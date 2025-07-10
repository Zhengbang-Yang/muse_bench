from .utils import load_model_and_tokenizer, load_model
from .dataset import ForgetRetainDataset
from torch import nn
import torch
import torch.nn.functional as F
from torch.cuda import device_count
import transformers
from transformers import Trainer, AutoModelForCausalLM


def unlearn(
    model_dir: str,
    data_file: str,
    out_dir: str,
    retain_data_file: str | None = None,
    loss_type: str = 'ga',
    per_device_batch_size: int = 2,
    gradient_accumulation_steps: int = 1,
    epochs: int = 5,
    learning_rate=1e-5,
    max_len: int = 4096,
    tokenizer_dir: str | None = None,
    resume_from_checkpoint: bool = False,
    report_to: str = 'none',  # Disable wandb by default
    logging_steps: int = 50,
    k: int=5,
    push_to_hub: bool=False,                   
    hub_model_id: str="",
):
    if 'gd' in loss_type:
        assert retain_data_file is not None, "Retain data must be specified for grad_diff."

    model, tokenizer = load_model_and_tokenizer(
        model_dir,
        tokenizer_dir=tokenizer_dir
    )

    ref_model = (
        load_model(model_dir)
        if 'npo' in loss_type or 'kl' in loss_type or 'ref' in loss_type
        else None
    )

    dataset = ForgetRetainDataset(
        data_file,
        tokenizer=tokenizer,
        retain_file_path=retain_data_file,
        max_len=max_len
    )

    if device_count() == 0:
        raise ValueError("Device not detected!")

    training_args = transformers.TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        save_strategy='no', 
        num_train_epochs=epochs,
        optim='adamw_torch',
        lr_scheduler_type='constant',
        bf16=True,
        report_to=report_to,        # none for Disable wandb
        logging_steps=logging_steps,
        push_to_hub=push_to_hub,                   
        hub_model_id=hub_model_id,  
    )

    trainer = IterativeUnlearner(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        data_collator=dataset.get_collate_fn(),
        loss_type=loss_type,
        k=k
    )
    model.config.use_cache = False  # silence the warnings.
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(out_dir)



class IterativeUnlearner(Trainer):
    """Source: https://github.com/locuslab/tofu/blob/main/dataloader.py
    """

    def __init__(self, *args,
                 loss_type: str = 'ga',
                 ref_model: AutoModelForCausalLM | None = None,
                 beta: float = 0.1,
                 k: int = 5,
                 **kwargs):
        self.loss_type = loss_type
        self.ref_model = ref_model
        self.beta = beta    # Only relevant when `'po' in self.loss_type`
        self.k=k
        print(self.loss_type)
        if ref_model is not None:
            assert 'po' in self.loss_type or 'kl' in self.loss_type or 'ref' in self.loss_type
            ref_model = ref_model.eval()

        super().__init__(*args, **kwargs)

    def save_model(self, output_dir=None, _internal_call=False):
        if self.args.push_to_hub:
            self.push_to_hub()

    def get_batch_loss(self, logits, labels):
        shifted_labels = labels[..., 1:].contiguous()
        logits = logits[..., :-1, :].contiguous()

        loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
        # get the sum loss for each sequence in a batch
        loss = loss_function(logits.transpose(-1,-2), shifted_labels).sum(dim=-1)

        return loss

    def compute_loss(self, model, x, return_outputs=False,num_items_in_batch: int | None = None):
        """Source: https://github.com/licong-lin/negative-preference-optimization/blob/main/synthetic/mymodel.py
        """
        
        ### 1. Run model ###
        x_f, x_r = x
        outputs_f = model(
            x_f['input_ids'],
            labels=x_f['labels'] if 'labels' in x_f else x_f['input_ids'].clone(),
            attention_mask=x_f['attention_mask'] if 'attention_mask' in x_f else torch.ones_like(x_f['input_ids'], dtype=torch.bool)
        )
        loss_f = outputs_f.loss

        if 'gdr' in self.loss_type or 'klr' in self.loss_type:
            outputs_r = model(
                x_r['input_ids'],
                labels=x_r['labels'] if 'labels' in x_r else x_r['input_ids'].clone(),
                attention_mask=x_r['attention_mask'] if 'attention_mask' in x_r else torch.ones_like(x_r['input_ids'], dtype=torch.bool)
            )
            loss_r = outputs_r.loss

        if 'klf' in self.loss_type or 'npo' in self.loss_type or 'ref' in self.loss_type:
            with torch.no_grad():
                outputs_f_ref = self.ref_model(
                    x_f['input_ids'],
                    labels=x_f['labels'] if 'labels' in x_f else x_f['input_ids'].clone(),
                    attention_mask=x_f['attention_mask'] if 'attention_mask' in x_f else torch.ones_like(x_f['input_ids'], dtype=torch.bool)
                )

        if 'klr' in self.loss_type:
            with torch.no_grad():
                outputs_r_ref = self.ref_model(
                    x_r['input_ids'],
                    labels=x_r['labels'] if 'labels' in x_r else x_r['input_ids'].clone(),
                    attention_mask=x_r['attention_mask'] if 'attention_mask' in x_r else torch.ones_like(x_r['input_ids'], dtype=torch.bool)
                )

        ### 2. Compute Loss ###
        loss = 0

        if 'npo_gather_shift' in self.loss_type:
            neg_log_p=self.get_batch_loss(outputs_f.logits, x_f['labels'])
            neg_log_p_ref=self.get_batch_loss(outputs_f_ref.logits, x_f['labels'])
            neg_log_ratio = neg_log_p - neg_log_p_ref
            loss += -F.logsigmoid(self.beta * neg_log_ratio).mean() * 2 / self.beta
            self.log({"pi(y|x)": torch.exp(-neg_log_p).mean().item()})
            self.log({"pi^(y|x)": torch.exp(-neg_log_p_ref).mean().item()})

        elif 'TWISE_gather_shift' in self.loss_type:
            neg_log_p=self.get_batch_loss(outputs_f.logits, x_f['labels'])
            with torch.no_grad():
                neg_log_p_hat  = self.get_batch_loss(outputs_f.logits, x_f['labels'])
            log_1m_p_hat = torch.log1p(-torch.exp(-neg_log_p_hat).clamp_max(1-1e-7))  
            neg_log_prob_ratio = log_1m_p_hat + neg_log_p
            loss += -F.logsigmoid(self.beta * neg_log_prob_ratio).mean() * 1 / self.beta
            print(f"Step {self.state.global_step}: pi(y|x) = {torch.exp(neg_log_p).mean().item()}")
            self.log({"pi(y|x)": torch.exp(neg_log_p).mean().item()})
            self.log({"1-pi^(y|x)": torch.exp(log_1m_p_hat).mean().item()})

        elif 'TWISE_gather' in self.loss_type:
            log_p  = F.log_softmax(outputs_f.logits,  dim=-1)      # log π(y|x)
            log_p = log_p.gather(dim=2, index=x_f['labels'].unsqueeze(-1)).squeeze(-1)
            with torch.no_grad():
                log_p_hat  = F.log_softmax(outputs_f.logits,  dim=-1)      # log π^(y|x) 
                log_p_hat = log_p_hat.gather(dim=2, index=x_f['labels'].unsqueeze(-1)).squeeze(-1)
            log_1m_p_hat = torch.log1p(-torch.exp(log_p_hat).clamp_max(1-1e-7))  
            neg_log_prob_ratio = log_1m_p_hat - log_p
            loss += -F.logsigmoid(self.beta * neg_log_prob_ratio).mean() * 1 / self.beta
            print(f"Step {self.state.global_step}: pi(y|x) = {torch.exp(log_p).mean().item()}")
            self.log({"pi(y|x)": torch.exp(log_p).mean().item()})
            self.log({"1-pi^(y|x)": torch.exp(log_1m_p_hat).mean().item()})

        elif 'TWISE_top_ref' in self.loss_type and 'ref' in self.loss_type:
            logits_ref = outputs_f_ref.logits  # [B, S, V]
            labels_ref = x_f['labels'] if 'labels' in x_f else x_f['input_ids'].clone()
            
            # Top-5 logits and their indices along vocab dimension
            top5_logits_ref, top5_indices_ref = torch.topk(logits_ref, k=5, dim=-1)  # both: [B, S, 5]
            
            # 为了收集每个 label 在 logits 中对应的值
            batch_size, seq_len = labels_ref.shape
            label_logits_ref = torch.gather(logits_ref, 2, labels_ref.unsqueeze(-1)).squeeze(-1)  # [B, S]
            
            # 拼接：每个位置的 top-5 + label 对应的 logit
            # 为了能拼接，我们先扩展 label_logits 的维度
            label_logits_unsq_ref = label_logits_ref.unsqueeze(-1)  # [B, S, 1]
            
            # 检查 label 是否已在 top-5 中（如果是，可以选择是否去重）
            # 这里我们无条件拼接，保留 top-5 + label（共6个logits）
            extended_logits_ref = torch.cat([top5_logits_ref, label_logits_unsq_ref], dim=-1)  # [B, S, 6]

            logits_f = outputs_f.logits  # [B, S, V]
            labels_f = x_f['labels'] if 'labels' in x_f else x_f['input_ids'].clone()
            top5_logits_f = torch.gather(logits_f, dim=2, index=top5_indices_ref)
            label_logits_f = torch.gather(logits_f, 2, labels_f.unsqueeze(-1)).squeeze(-1)  # [B, S]
            label_logits_unsq_f = label_logits_f.unsqueeze(-1)
            extended_logits_f = torch.cat([top5_logits_f, label_logits_unsq_f], dim=-1)  # [B, S, 6]
            
            log_p  = F.log_softmax(extended_logits_f,  dim=-1)[..., :-1, :]      # log π(y|x)
            log_p_ref  = F.log_softmax(extended_logits_ref,  dim=-1)[..., :-1, :]      # log π(y|x)
            neg_log_prob_ratio = log_p_ref - log_p
            if "ln" in self.loss_type:
                loss += -F.logsigmoid(self.beta / outputs_f.logits.shape[1] * neg_log_prob_ratio).mean() * 1 / self.beta
            else:    
                loss += -F.logsigmoid(self.beta * neg_log_prob_ratio).mean() * 1 / self.beta
            self.log({"pi(y|x)": torch.exp(log_p).mean().item()})
            self.log({"pi_ref(y|x)": torch.exp(log_p_ref).mean().item()})
        
        elif 'TWISE_top' in self.loss_type:
            logits = outputs_f.logits  # [B, S, V]
            labels = x_f['labels'] if 'labels' in x_f else x_f['input_ids'].clone()
            
            # Top-5 logits and their indices along vocab dimension
            top5_logits, top5_indices = torch.topk(logits, k=5, dim=-1)  # both: [B, S, 5]
            
            # 为了收集每个 label 在 logits 中对应的值
            batch_size, seq_len = labels.shape
            label_logits = torch.gather(logits, 2, labels.unsqueeze(-1)).squeeze(-1)  # [B, S]
            
            # 拼接：每个位置的 top-5 + label 对应的 logit
            # 为了能拼接，我们先扩展 label_logits 的维度
            label_logits_unsq = label_logits.unsqueeze(-1)  # [B, S, 1]
            
            # 检查 label 是否已在 top-5 中（如果是，可以选择是否去重）
            # 这里我们无条件拼接，保留 top-5 + label（共6个logits）
            extended_logits = torch.cat([top5_logits, label_logits_unsq], dim=-1)  # [B, S, 6]
            
            # logits = outputs_f.logits
            # top5_logits, top5_indices = torch.topk(outputs_f.logits, k=self.k, dim=-1)
            # labels=x_f['labels']
            # labels_unsq = labels.unsqueeze(-1)  # [batch_size, seq_len, 1]
            # in_top5 = (top5_indices == labels_unsq).any(dim=-1)  # [batch_size, seq_len]
            # final_logits = []
            # for b in range(logits.size(0)):
            #     batch_logits = []
            #     for t in range(logits.size(1)):
            #         top5 = top5_logits[b, t]
            #         top5_ids = top5_indices[b, t]
            #         label_id = labels[b, t].item()
            
            #         batch_logits.append(top5)
            #         label_logit = logits[b, t, label_id].unsqueeze(0)
            #         extended_logits = torch.cat([top5, label_logit], dim=0)
            #         batch_logits.append(extended_logits)
            #     final_logits.append(torch.stack(, dim=0))
            # final_logits = torch.stack(final_logits, dim=0)
            
            log_p  = F.log_softmax(extended_logits,  dim=-1)[..., :-1, :]      # log π(y|x)
            with torch.no_grad():
                log_p_hat  = F.log_softmax(extended_logits,  dim=-1)[..., :-1, :]      # log π^(y|x)                 
            log_1m_p_hat = torch.log1p(-torch.exp(log_p_hat).clamp_max(1-1e-7))  
            neg_log_prob_ratio = log_1m_p_hat - log_p
            if "ln" in self.loss_type:
                loss += -F.logsigmoid(self.beta / outputs_f.logits.shape[1] * neg_log_prob_ratio).mean() * 1 / self.beta
            else:    
                loss += -F.logsigmoid(self.beta * neg_log_prob_ratio).mean() * 1 / self.beta
            self.log({"pi(y|x)": torch.exp(log_p).mean().item()})
            self.log({"1-pi^(y|x)": torch.exp(log_1m_p_hat).mean().item()})
        elif 'npo_top' in self.loss_type:
            neg_log_ratio = outputs_f_ref.logits - outputs_f.logits
            loss += -F.logsigmoid(self.beta * neg_log_ratio).mean() * 2 / self.beta
        elif 'ga' in self.loss_type:
            loss += -loss_f

        else:
            raise NotImplementedError("Cannot infer the given loss type.")

        if 'gdr' in self.loss_type:
            loss += loss_r
            self.log({"gd": loss_r.mean().item()})

        if 'klf' in self.loss_type:
            raise NotImplementedError("KL forget not implemented yet!")

        if 'klr' in self.loss_type:
            kl_r = F.kl_div(
                F.log_softmax(outputs_r.logits,  dim=-1),
                F.log_softmax(outputs_r_ref.logits, dim=-1),
                # outputs_r.logits,
                # outputs_r_ref.logits,
                reduction = 'batchmean',
                log_target = True
            )
            loss += kl_r
            self.log({"kl": kl_r.mean().item()})

        return (loss, outputs_f) if return_outputs else loss


    def prediction_step(self, model, x, prediction_loss_only: bool, ignore_keys=None):
        input_ids, labels, attention_mask = x
        # forward pass
        with torch.no_grad():
            outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
            logits = outputs.logits
            loss = outputs.loss
        return (loss, logits, labels)

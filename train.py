

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from trainer import SpeechLLMLightning
from dataset import VLSPDataset, MyCollator  # Remove DataLoader from here
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader  # Add this import
import torch.utils.data as data_utils
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import wandb
from safetensors.torch import save_file
import os
from pytorch_lightning.callbacks import Callback

class SafetensorsSaveCallback(Callback):
    def __init__(self, save_dir, model_name):
        super().__init__()
        self.save_dir = save_dir
        self.model_name = model_name
        self.best_val_loss = float('inf')
        os.makedirs(save_dir, exist_ok=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        val_loss = trainer.callback_metrics.get('val/wer')
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
        
            state_dict = {
                'audio_encoder': pl_module.audio_encoder.state_dict(),
                'connector': pl_module.connector.state_dict(),
                'llm_model': pl_module.llm_model.state_dict()
            }
            
            # Save using safetensors
            save_path = os.path.join(
                self.save_dir, 
                f"{self.model_name}_epoch{trainer.current_epoch}_loss{val_loss:.4f}.safetensors"
            )
            save_file(state_dict, save_path)
            print(f"Saved best model to {save_path}")

if __name__ == "__main__":
    log_path = 'whisper-linear-qwen0.5-run1'
    wandb.init(project="mmllm", name=log_path)
    logger = WandbLogger(project="mmllm", name=log_path)

    model_config = {
                'audio_enc_dim': 1024, 
                'llm_dim': 896, 
                'audio_encoder_name': "minicpm-o", 
                'connector_name': 'conformer',
                'llm_name': "/kaggle/input/qwen2.5/transformers/0.5b/1",
                'finetune_encoder': False,
                'finetune_llm': True,
                'finetune_connector': False,
                'connector_k': 2,
                'use_lora': False,
                'lora_r': 8,
                'lora_alpha': 16,
                'max_lr': 1e-4,
                'total_training_step': 10000000,
                'warmup_steps': 100,
                'train_batch_per_epoch': 10000,
                'grad_accumulate_steps': 8
        }   
    
    model = SpeechLLMLightning(**model_config)
    print(model)
    tokenizer = model.llm_tokenizer

    # train_dataset = InstructionalAudioDataset(
    #     csv_file = './data_samples/train.csv',
    #     mode='train', 
    #     random_keys_prob=0.2,
    #     )

    # val_dataset = InstructionalAudioDataset(
    #     csv_file='./data_samples/dev.csv', 
    #     mode='test'
    #     )

    # print(len(train_dataset), len(val_dataset))

    # my_collator = MyCollator(model_config['audio_encoder_name'], tokenizer)
    # train_loader = data_utils.DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=my_collator, num_workers=3)
    # val_loader = data_utils.DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=my_collator, num_workers=3)


    train_dataset = VLSPDataset(split="train", streaming=True)
    val_dataset = VLSPDataset(split="validation", streaming=True)

    # Initialize data loaders
    my_collator = MyCollator(model_config['audio_encoder_name'], tokenizer)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=1,
        num_workers=3,
        collate_fn=my_collator
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=1,
        num_workers=3,
        collate_fn=my_collator
    )

    checkpoint_callback = ModelCheckpoint(dirpath=f"checkpoints", filename=log_path+'-{epoch}', save_top_k=1, monitor="val/wer", save_last=True)
    safetensors_callback = SafetensorsSaveCallback(
        save_dir="safetensors_checkpoints",
        model_name=log_path
    )
    early_stop_callback = EarlyStopping(monitor="val/loss", min_delta=0.00, patience=10, verbose=False, mode="min")

    trainer = Trainer(
            max_epochs=model_config['total_training_step']//model_config['train_batch_per_epoch'], 
            strategy=DDPStrategy(find_unused_parameters=True),
            limit_train_batches=model_config['train_batch_per_epoch'], 
            limit_val_batches=model_config['train_batch_per_epoch'], 
            log_every_n_steps=model_config['train_batch_per_epoch'], 
            enable_checkpointing=True, 
            callbacks=[checkpoint_callback, safetensors_callback],
            fast_dev_run=False, logger=logger, 
            accumulate_grad_batches=model_config['grad_accumulate_steps'],
    )
    trainer.fit(model, train_loader, val_loader)


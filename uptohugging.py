from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("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"))
api.upload_folder(
    folder_path="E:\Vinbigdata\EOT\whisper_streaming",
    repo_id="NMCxyz/whisper_streaming",
    repo_type="model",
)

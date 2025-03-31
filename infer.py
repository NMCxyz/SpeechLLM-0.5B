import torch
import sys
import json
import librosa
from transformers import AutoProcessor, AutoConfig, LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, Qwen2ForCausalLM, Qwen2Tokenizer
import numpy as np
sys.path.append("/mnt/4T2/thuctap/cuongnm75/MultimodalVisualVideoStreaming/EOT/whisper_streaming/speech_encoder")
from minicpmo_whisper import MiniCPMO_Whisper, MiniCPMWhisperEncoder, WhisperConfig
from model.connector import get_connector


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

class SpeechLLMInference:
    def __init__(self):

        self.minipcmo_config = json.load(open("/mnt/4T2/thuctap/cuongnm75/MultimodalVisualVideoStreaming/EOT/whisper_streaming/infer/minicpmp_config.json", "r"))
        audio_config = json.load(open("/mnt/4T2/thuctap/cuongnm75/MultimodalVisualVideoStreaming/EOT/whisper_streaming/infer/audio_config.json", "r"))
        self.audio_config = WhisperConfig(**audio_config)
        self.apm_path = "/mnt/4T2/thuctap/cuongnm75/MultimodalVisualVideoStreaming/EOT/whisper_streaming/speech_encoder/weights/apm_weights.pth"
        self.avg_pooler_path = "/mnt/4T2/thuctap/cuongnm75/MultimodalVisualVideoStreaming/EOT/whisper_streaming/speech_encoder/weights/audio_avg_pooler_weights.pth"
        self.projection_path = "/mnt/4T2/thuctap/cuongnm75/MultimodalVisualVideoStreaming/EOT/whisper_streaming/speech_encoder/weights/audio_projection_layer_weights.pth"
        
        self.model_config = {
            'audio_enc_dim': 3584,
            'llm_dim': 896,  
            'connector_name': 'conformer',
            'connector_k': 2
        }
        
        self.audio_encoder = MiniCPMWhisperEncoder(self.audio_config)
        print(self.audio_encoder)

        self.audio_model = MiniCPMO_Whisper(
            self.minipcmo_config, 
            audio_config,
            self.apm_path,
            self.avg_pooler_path,
            self.projection_path,
            torch_dtype=torch.bfloat16
        )
        print(self.audio_model)
        
        self.connector = get_connector(
            self.model_config['connector_name'],
            self.model_config['audio_enc_dim'],
            self.model_config['llm_dim'],
            self.model_config['connector_k']
        ).to(dtype=torch.bfloat16)


        print(f"\nConnector parameters: {count_parameters(self.connector):,}")

        # llm_path = "/mnt/4T2/thuctap/cuongnm75/MultimodalVisualVideoStreaming/EOT/turn-detector"
        llm_path = "/mnt/4T2/thuctap/cuongnm75/MultimodalVisualVideoStreaming/Reference/MiniCPM-o/model/Qwen2.5-0.5B-Instruct"
        # self.llm_tokenizer = AutoTokenizer.from_pretrained(
        #     llm_path,
        #     trust_remote_code=True,
        #     use_fast=False
        # )
        # self.llm_model = LlamaForCausalLM.from_pretrained(
        #     llm_path,
        #     torch_dtype=torch.bfloat16,
        #     device_map="auto"
        # )
        self.llm_tokenizer = Qwen2Tokenizer.from_pretrained(
            llm_path,
            trust_remote_code=True,
            use_fast=False
        )
        self.llm_model = Qwen2ForCausalLM.from_pretrained(
            llm_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

        print(f"LLM parameters: {count_parameters(self.llm_model):,}\n")

        self.device = torch.device('cuda')
        self.audio_encoder = self.audio_encoder.eval().cuda()
        self.connector = self.connector.to(self.device).eval()
        
        self.processor = AutoProcessor.from_pretrained(
            "/mnt/4T2/thuctap/cuongnm75/MultimodalVisualVideoStreaming/Reference/MiniCPM-o/train/minicpmo",
            trust_remote_code=True
        )

    # def process_audio(self, audio_path):
    #     audio_input, _ = librosa.load(audio_path, sr=16000, mono=True)
    #     audios = [[audio_input]]
    #     audio_parts = [[1]]
        
    #     audio_features, audio_feature_lens, _ = self.processor.audio_feature_extract(
    #         audios, audio_parts, chunk_input=True, sampling_rate=16000
    #     )

    #     data = {
    #         "audio_features": audio_features,
    #         "audio_feature_lens": audio_feature_lens
    #     }

    #     with torch.no_grad():
    #         res = self.audio_encoder.get_audio_embedding(data, chunk_length=1)
    #         speech_embeds = res[0][0].to(self.device)
    #         speech_embeds = self.connector(speech_embeds.unsqueeze(0))

    #     pre_prompt = "Instruction:\nĐừng đọc hiểu âm thanh, hãy nói 1 câu tiếng Việt\n\nInput:\n<speech>"
    #     post_prompt = "</speech>\n\nOutput:\n"

    #     # pre_tokenized = self.llm_tokenizer(pre_prompt, return_tensors='pt').input_ids.to(self.device)
    #     # post_tokenized = self.llm_tokenizer(post_prompt, return_tensors='pt').input_ids.to(self.device)

    #     # embedder = self.llm_model.model.embed_tokens
    #     # pre_embeds = embedder(pre_tokenized)
    #     # post_embeds = embedder(post_tokenized)
    #     # combined_embeds = torch.cat([pre_embeds,  speech_embeds, post_embeds], dim=1)

    #     # with torch.no_grad():
    #     #     generated_ids = self.llm_model.generate(
    #     #         inputs_embeds=combined_embeds,
    #     #         num_beams=3,
    #     #         early_stopping=True,
    #     #         temperature=0.9
    #     #     )
        
    #     # print(f"Generated IDs: {generated_ids}")
    #     # output = self.llm_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    #     # return output

    #         # Add debug prints for prompts
    #     # Chuẩn bị prompt với <speech> placeholder
    #     messages = [
    #         {
    #             "role": "system", 
    #             "content": "You are a helpful assistant that speaks Vietnamese."
    #         },
    #         {
    #             "role": "user", 
    #             "content": "Instruction:\nĐừng đọc hiểu âm thanh, hãy nói 1 bài thơ tiếng việt\n\nInput:\n<speech></speech>\n\nOutput:"
    #         }
    #     ]

    #     # Apply chat template
    #     text = self.llm_tokenizer.apply_chat_template(
    #         messages,
    #         tokenize=False,
    #         add_generation_prompt=True
    #     )
    #     print(f"Prompt template: {text}")
        
    #     # Tokenize text
    #     model_inputs = self.llm_tokenizer(
    #         text, 
    #         return_tensors="pt",
    #         add_special_tokens=True
    #     ).to(self.device)
        
    #     # Tìm vị trí chính xác của token <speech> và </speech>
    #     input_ids = model_inputs.input_ids[0].tolist()
    #     speech_start_tokens = self.llm_tokenizer.encode("<speech>", add_special_tokens=False)
    #     speech_end_tokens = self.llm_tokenizer.encode("</speech>", add_special_tokens=False)
        
    #     # Tìm vị trí bắt đầu của <speech> token
    #     for i in range(len(input_ids) - len(speech_start_tokens) + 1):
    #         if input_ids[i:i+len(speech_start_tokens)] == speech_start_tokens:
    #             speech_start_idx = i + len(speech_start_tokens)
    #             break
        
    #     # Tìm vị trí bắt đầu của </speech> token
    #     for i in range(speech_start_idx, len(input_ids) - len(speech_end_tokens) + 1):
    #         if input_ids[i:i+len(speech_end_tokens)] == speech_end_tokens:
    #             speech_end_idx = i
    #             break
        
    #     print(f"Speech token positions: start={speech_start_idx}, end={speech_end_idx}")
        
    #     # Lấy input embeddings từ mô hình
    #     input_embeds = self.llm_model.model.embed_tokens(model_inputs.input_ids)
        
    #     # Tạo embeddings mới bằng cách thay thế phần giữa <speech> và </speech> bằng speech_embeds
    #     new_embeds = torch.cat([
    #         input_embeds[:, :speech_start_idx],
    #         speech_embeds,  # Thêm speech embeddings
    #         input_embeds[:, speech_end_idx:]  # Tiếp tục từ sau </speech>
    #     ], dim=1)
        
    #     # Tính toán attention mask mới
    #     original_length = model_inputs.attention_mask.size(1)
    #     speech_token_count = speech_end_idx - speech_start_idx  # Số token giữa <speech> và </speech>
    #     embedding_token_count = speech_embeds.size(1)  # Số token của speech embedding
    #     new_length = original_length - speech_token_count + embedding_token_count
        
    #     # Tạo attention mask mới với kích thước phù hợp
    #     new_attention_mask = torch.ones((1, new_length), dtype=torch.long, device=self.device)
        
    #     # Generate với embeddings đã chỉnh sửa
    #     with torch.no_grad():
    #         outputs = self.llm_model.generate(
    #             inputs_embeds=new_embeds,
    #             attention_mask=new_attention_mask,
    #             max_new_tokens=512,
    #             num_beams=3,
    #             early_stopping=True,
    #             temperature=0.7,
    #             repetition_penalty=1.2,
    #             no_repeat_ngram_size=3,
    #             pad_token_id=self.llm_tokenizer.pad_token_id,
    #             bos_token_id=self.llm_tokenizer.bos_token_id,
    #             eos_token_id=self.llm_tokenizer.eos_token_id,
    #             return_dict_in_generate=True,
    #             output_scores=False
    #         )
        
    #     # Lấy các token đã sinh ra
    #     generated_ids = outputs.sequences
        
    #     # Tính toán độ dài của context để chỉ giữ lại phần mới sinh ra
    #     # Do chúng ta không có model_inputs.input_ids chính xác cho new_embeds,
    #     # chúng ta sẽ dùng input_length_with_speech_embed
    #     input_length_with_speech_embed = new_length
        
    #     # Trích xuất chỉ phần mới được sinh ra
    #     new_tokens = generated_ids[0][input_length_with_speech_embed:]
        
    #     # Giải mã kết quả
    #     response = self.llm_tokenizer.decode(new_tokens, skip_special_tokens=True)
        
    #     return response.strip()



    def process_audio(self, audio_path):
        # Process audio
        audio_input, _ = librosa.load(audio_path, sr=16000, mono=True)
        audios = [[audio_input]]
        audio_parts = [[1]]
        
        # Get audio features
        audio_features, audio_feature_lens, _ = self.processor.audio_feature_extract(
            audios, audio_parts, chunk_input=True, sampling_rate=16000
        )
        data = {
            "audio_features": audio_features,
            "audio_feature_lens": audio_feature_lens
        }

        # Get speech embeddings through encoder and connector
        with torch.no_grad():
            res = self.audio_model.get_audio_embedding(data, chunk_length=1)
            speech_embeds = res[0][0].to(self.device)
            # speech_embeds = self.connector(speech_embeds.unsqueeze(0))
            print(f"Speech embeddings shape: {speech_embeds.shape}")

        # Prepare chat messages
        messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that speaks Vietnamese."
                },
                {
                    "role": "user",
                    "content": "Hãy viết một bài thơ tiếng Việt"
                }
            ]

        # Apply chat template like test_eot_model
        text = self.llm_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Get input embeddings like trainer
        model_inputs = self.llm_tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=True
        ).to(self.device)

        # Get embeddings from LLM
        embedder = self.llm_model.model.embed_tokens
        text_embeds = embedder(model_inputs.input_ids)

        # Combine embeddings like trainer
        combined_embeds = torch.cat([text_embeds, speech_embeds], dim=1)
        print(f"Combined embeddings shape: {combined_embeds.shape}")

        # Create attention mask
        attention_mask = torch.ones(
            (combined_embeds.shape[0], combined_embeds.shape[1]),
            dtype=torch.long,
            device=self.device
        )

        # Generate like test_eot_model
        with torch.no_grad():
            outputs = self.llm_model.generate(
                inputs_embeds=combined_embeds,
                attention_mask=attention_mask,
                max_new_tokens=512,
                num_beams=3,
                early_stopping=True,
                temperature=0.7,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                pad_token_id=self.llm_tokenizer.pad_token_id,
                bos_token_id=self.llm_tokenizer.bos_token_id,
                eos_token_id=self.llm_tokenizer.eos_token_id,
                return_dict_in_generate=True
            )
        # print(outputs)
        # Get new tokens only
        generated_ids = outputs
        input_length = combined_embeds.shape[1]
        new_tokens = generated_ids[0][input_length:]
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        # Decode response
        response = self.llm_tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        print("\nGeneration details:")
        print(f"Input length: {input_length}")
        print(f"Generated length: {len(generated_ids[0])}")
        print(f"New tokens: {len(new_tokens)}")

        return response.strip()
    

if __name__ == "__main__":
    model = SpeechLLMInference()
    print(model)
    audio_path = "/mnt/4T2/thuctap/cuongnm75/MultimodalVisualVideoStreaming/EOT/vietnamese-eot-finetune/gen_tts_sample/smartdata4_audio/converted_sample_3.wav"
    result = model.process_audio(audio_path)
    print("\n===== PREDICTION RESULTS =====")
    print(f"Audio file: {audio_path}")
    print(f"Model output: {result}")
from openai import OpenAI
import yaml
import re
from argparse import Namespace
import base64
from PIL import Image
from typing import List, Dict
from transformers import (
    CLIPProcessor, CLIPModel
)
import torchvision.transforms as T
import cv2
import numpy as np
import json
import torch
import os
from omegaconf import OmegaConf
import argparse
from pathlib import Path
import io

from reward_model import SimpleNN

# Load CLIP
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

class PromptPilot:
    def __init__(self, seine_model, device="cuda"):
        self.llm = OpenAI(
                api_key="fy3jHNMV7OC7t7fQprQkFgp7NeSlRsMG",
                base_url="https://api.deepinfra.com/v1/openai",
            )
        self.temperature = 0
        self.max_iterations = 2
        self.seine_model = seine_model
        self.device = device

        self.SYSTEM_PROMPT = """
                                You are a prompt refinement agent specialized in enhancing prompts for video generation models that take a single image as input.
                                Your goal is to improve the given prompt so that the generated video is visually coherent, smooth in motion, and logically consistent with the content and context of the image.
                                Carefully consider the visual elements and implied actions in the image, and rewrite the prompt to guide the model toward generating a realistic and temporally logical video sequence.
                             """

        self.USER_PROMPT = """
                              Given an image and a history of previous prompts with their corresponding scores for their generated videos, refine and rewrite the prompt to  further enhance the video quality. Ensure that the refined prompt is an improved version of the previous prompts, providing more guidance as appropriate.

                              Output the refined prompt in the following format:
                              Refined prompt: <refined prompt>

                              If you are not able to refine the prompt, output the following:
                              Refined prompt: <previous prompt>

                           """

        self.messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]

        self.resize = T.Resize((224, 224))
        self.to_tensor = T.ToTensor()
        self.scores = {
            "clip_tva_score": 0,
            "temporal_consistency": 0,
            "dynamic_degree": 0}

        self.reward_model = SimpleNN()
        self.reward_model.load_state_dict(torch.load('user_marks/model/reward_model_state_dict_NN.pth'))
        self.reward_model.eval()

    def extract_frames(self,video_path: str, num_frames: int = 4) -> List[Image.Image]:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idxs = np.linspace(0, total_frames - 1, num_frames).astype(int)

        frames = []
        for idx in frame_idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(rgb_frame))
        cap.release()
        return frames

    def compute_clip_alignment(self, frames: List[Image.Image], prompt: str) -> float:
        # runcate prompt to CLIP's max token limit (77)
        truncated_prompt = clip_processor.tokenizer.decode(
            clip_processor.tokenizer(prompt, truncation=True, max_length=77)["input_ids"],
            skip_special_tokens=True
        )
        inputs = clip_processor(
            text=[truncated_prompt] * len(frames), images=frames,
            return_tensors="pt", padding=True
        ).to(self.device)
        with torch.no_grad():
            outputs = clip_model(**inputs)
            sims = torch.cosine_similarity(outputs.image_embeds, outputs.text_embeds)
        return sims.mean().item()
    
    def compute_temporal_consistency(self, frames: List[Image.Image]) -> float:
        gray_frames = [cv2.cvtColor(np.array(f), cv2.COLOR_RGB2GRAY) for f in frames]
        total_flow = 0.0
        for i in range(1, len(gray_frames)):
            flow = cv2.calcOpticalFlowFarneback(
                gray_frames[i - 1], gray_frames[i], None,
                pyr_scale=0.5, levels=3, winsize=15, iterations=3,
                poly_n=5, poly_sigma=1.2, flags=0
            )
            magnitude = np.linalg.norm(flow, axis=2).mean()
            total_flow += magnitude
        return total_flow / (len(frames) - 1)

    def compute_dynamic_degree(self, frames: List[Image.Image]) -> float:
        """
        Computes dynamic degree as the variance of frame-to-frame pixel differences.
        Higher values imply more movement or dynamic content.
        """
        gray_frames = [cv2.cvtColor(np.array(f), cv2.COLOR_RGB2GRAY) for f in frames]
        diffs = []

        for i in range(1, len(gray_frames)):
            diff = np.abs(gray_frames[i].astype(np.float32) - gray_frames[i - 1].astype(np.float32))
            mean_diff = diff.mean()
            diffs.append(mean_diff)

        return np.var(diffs) if diffs else 0.0

    def evaluate_video(self, frames: List[Image.Image], prompt: str) -> Dict[str, float]:
        # print("frames",frames)
        clip_score = self.compute_clip_alignment(frames, prompt)
        tc_score = self.compute_temporal_consistency(frames)
        dd_score = self.compute_dynamic_degree(frames)

        self.scores.update({
            "clip_tva_score": clip_score,
            "temporal_consistency": tc_score,
            "dynamic_degree": dd_score
        })

        return self.scores  # Return updated dictionary if needed
    
    def set_first_text_prompt(self, yaml_path):
        with open(yaml_path, 'r') as f:
            content = yaml.safe_load(f)

        # Extract the input_path value to use as the new_prompt
        input_path = content.get("input_path", "No input_path found")
        prompt = input_path.split('/')[-1].split('.')[0].replace('_', ' ')
    
        with open(yaml_path, 'r') as f:
            content = f.read()

        # replace the text_prompt line
        new_content = re.sub(
            r'text_prompt:.*?\n',
            f'text_prompt: ["{prompt}"]\n',
            content
        )

        with open(yaml_path, 'w') as f:
            f.write(new_content)

    def update_text_prompt(self, yaml_path, refined_prompt):
        with open(yaml_path, 'r') as f:
            content = f.read()

        # replace the text_prompt line
        new_content = re.sub(
            r'text_prompt:.*?\n',
            f'text_prompt: ["{refined_prompt}"]\n',
            content
        )

        with open(yaml_path, 'w') as f:
            f.write(new_content)

    def query_llm(self):
        response = self.llm.chat.completions.create(
            model="meta-llama/Llama-3.2-90B-Vision-Instruct",
            messages=self.messages,
            temperature=self.temperature
        )
        return response.choices[0].message.content

    def clean_response(self, response):
        refined_match = re.search(r"Refined prompt:\s*(.*?\.)(\s|$)", response)

        refined_prompt = refined_match.group(1).strip() if refined_match else ""

        return refined_prompt
    
    def cleanup(self):
        self.messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        torch.cuda.empty_cache()

    def infer(self, image, prompt):
        print("Generating video...")
        
        torch.cuda.empty_cache()
        
        save_path = Path(f"./results/generated_videos/{prompt}")
        save_path.mkdir(parents=True, exist_ok=True)

        # prepare llm image prompt
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        encoded_image = base64.b64encode(buffer.read()).decode("utf-8")
        
        self.messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64, {encoded_image}"
                            }
                        }
                    ]
                }
            )

        clip_score = tc_score = dd_score = human_score = None
        best_clip_score = best_tc_score = best_dd_score = best_human_score = None
        best_video = best_prompt = None
        history = ""
        current_prompt = prompt
        
        iteration = 0
        while iteration < self.max_iterations and (not human_score or human_score < 7.0):
            print(f"\nIteration {iteration+1}/{self.max_iterations}")
            print("Current prompt:", current_prompt)

            # prepare llm text prompt
            if history == "":
                history += f"""
                                Previous prompt: {current_prompt}
                            """

                text = self.USER_PROMPT + \
                        f"""
                            {history}

                            Refined prompt:
                        """

            else:
                history += f"""
                                Previous prompt: {current_prompt}
                                CLIP Alignment score: {clip_score}
                                Temporal Consistency score: {tc_score}
                                Dynamic Degree score: {dd_score}
                                Human score: {human_score}
                            """
                
                text = f"""
                            The video was evaluated using four metrics:
                            1. CLIP Alignment Score, which measures how well the generated video frames align with the given text prompt. If CLIP Alignment is low, clarify key objects, actions or settings to help visuals match the description more precisely.
                            2. Temporal Consistency Score, which assesses the smoothness and coherence between consecutive video frames. If Temporal Consistency is low, simplify actions or reduce ambiguity that might lead to jittery or incoherent motion.
                            3. Dynamic Degree Score quantifies the amount of motion in the video by comparing pixel changes across frames. If Dynamic Degree is too high for a static scene, remove unnecessary verbs or motion elements. If Dynamic Degree is too low for an action scene, enhance verbs or scene dynamics. For example, "a cat walking" → "a cat jumping between rooftops".
                            4. Human Score is the aggregated score between 0-10 for overall quality. If Human Score is low, balance all factors with clearer, more focused language.
                        """ + \
                        self.USER_PROMPT + \
                        f"""
                            {history}

                            Refined prompt:
                        """

            # create llm use prompt
            self.messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": text
                        }
                    ]
                }
            )

            # get llm response
            response = self.query_llm()
            print(response)

            # clean up response
            refined_prompt = self.clean_response(response)

            if refined_prompt == current_prompt:
                break

            # prepare args for video generation
            video_path = self.seine_model.generate_video(save_path, image, refined_prompt)

            # compute evaluation metrics
            frames = self.extract_frames(video_path)
            scores = self.evaluate_video(frames, refined_prompt)
            
            # get aggregated score from reward model
            X = np.array(list(scores.values())[:3], dtype=np.float32)
            X = torch.tensor(X, dtype=torch.float32)
            scores["human_score"] = self.reward_model(X).detach().item()
            
            clip_score = scores["clip_tva_score"]
            tc_score = scores["temporal_consistency"]
            dd_score = scores["dynamic_degree"]
            human_score = scores["human_score"]
            print(f"Scores: {scores}")
            
            if best_human_score is None or human_score > best_human_score:
                best_video = video_path
                best_clip_score = clip_score
                best_tc_score = tc_score
                best_dd_score = dd_score
                best_human_score = human_score
                best_prompt = refined_prompt

            # Save to JSON
            result = {
                "prompt": refined_prompt,
                "scores": {k: float(v) for k, v in scores.items()}
            }
            json_path = f"{save_path}/scores.json"

            if os.path.exists(json_path):
                with open(json_path, "r") as f:
                    data = json.load(f)
            else:
                data = []

            data.append(result)

            with open(json_path, "w") as f:
                json.dump(data, f, indent=4)

            iteration += 1
            current_prompt = refined_prompt
            
        # cleanup after generation
        self.cleanup()

        return best_video, best_prompt, best_clip_score, best_tc_score, best_dd_score
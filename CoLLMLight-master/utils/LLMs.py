import requests
import json
import os
import time

    
class GPT_model:
    """OpenAI-compatible HTTP client (also supports Bailian)."""
    def __init__(self, llm_params=None, model="gpt-3.5-turbo", key=None, url=None,
                 timeout=120, max_attempts=2, use_env_proxy=True):
        base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.url = url or os.getenv("OPENAI_API_URL") or base + "/chat/completions"
        self.key = key or os.getenv("OPENAI_API_KEY")
        if not self.key:
            raise ValueError("Set OPENAI_API_KEY or pass key explicitly")
        authorization = self.key if self.key.lower().startswith("bearer ") else "Bearer " + self.key
        self.headers = {"Content-Type": "application/json", "Authorization": authorization}
        self.model = model
        self.llm_params = {"model": model, "temperature": 0.1, **(llm_params or {})}
        self.timeout = timeout
        self.session = requests.Session()
        self.session.trust_env = use_env_proxy
        self.max_retries = max_attempts
        self.total_consume = 0
        self.requests_count = 0
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def ask(self, user_prompt, system_prompt="You are a helpful AI assistant.", **kwargs):
        body = {"messages": [{"role": "system", "content": system_prompt},
                             {"role": "user", "content": user_prompt}], **self.llm_params}
        for attempt in range(self.max_retries):
            self.requests_count += 1
            try:
                response = self.session.post(self.url, headers=self.headers, json=body, timeout=self.timeout)
                if response.status_code in (400, 401, 403, 404, 422):
                    raise ValueError(f"Teacher API rejected configuration: HTTP {response.status_code}")
                response.raise_for_status()
                data = response.json()
                for name in self.usage:
                    self.usage[name] += int(data.get("usage", {}).get(name, 0))
                text = data["choices"][0]["message"]["content"]
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("Teacher returned empty content")
                return text
            except ValueError:
                raise
            except (requests.RequestException, KeyError, IndexError) as exc:
                if attempt + 1 == self.max_retries:
                    # Do not include request headers or provider response in errors.
                    raise RuntimeError(f"Teacher request failed after {self.max_retries} attempts ({type(exc).__name__})") from None
                time.sleep(min(2 ** attempt, 8))

    def batch_ask(self, user_prompts, system_prompt="You are a helpful AI assistant.", **kwargs):
        return [self.ask(p, system_prompt, **kwargs) for p in user_prompts]


class LLAMA_model:
    def __init__(self, model, llm_params = {}, port = 23333) -> None:
        self.url = "http://localhost:{}/v1/completions".format(port)
        self.headers = {
            "Content-Type": "application/json",
        }
        self.model = model
        self.session=requests.Session()
        self.session.trust_env=False
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model)
        self.llm_params = {
            "model": model,  
            "temperature": 0.1,
            # "prompt": prompt,
            "max_tokens": 8192,
            "stop_token_ids": [self.tokenizer.eos_token_id],
            "top_k": 50,
            "top_p": 0.95,
            "logprobs": 0,
        }
        self.llm_params.update(llm_params)

    def ask(self, user_prompt, system_prompt = 'You are a helpful AI assistant.', disable_thinking = False):
        messages=[  {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}]
        if disable_thinking:
            prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False
                )
        else:
            prompt = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )

        data = {
        "prompt": prompt,
        }
        data.update(self.llm_params)   
        try:
            response = self.session.post(self.url, headers=self.headers, data=json.dumps(data),timeout=120)
            response_text = response.json()['choices'][0]['text']
            return response_text
        except requests.exceptions.RequestException as e:
            return "Request failed"
        except json.JSONDecodeError as e:
            return "Error decoding JSON response"
        except KeyError as e:
            return "Key error"

    def batch_ask(self, user_prompts, system_prompt = 'You are a helpful AI assistant.', disable_thinking = False):
        messages_list = []
        batch_num = len(user_prompts)
        for user_prompt in user_prompts:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            if disable_thinking:
                prompt = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False
                    )
            else:
                prompt = self.tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True
                        )
            messages_list.append(prompt)

        data = {
            "prompt": messages_list,  
        }
        data.update(self.llm_params)   
        try:
            response = self.session.post(self.url, headers=self.headers, data=json.dumps(data),timeout=120)
            payload=response.json()
            response=sorted(payload['choices'],key=lambda c:c.get('index',0))
            self.last_generation_metadata=[{'completion_tokens':len((c.get('logprobs') or {}).get('tokens',[])),
                'token_count_source':'server_logprobs','finish_reason':c.get('finish_reason')} for c in response]
            response_text = [c['text'] for c in response]
            # print(response_text)
            return response_text
        except requests.exceptions.RequestException as e:
            # print(f"Request failed: {e}")
            return ["Request failed"] * batch_num
        except json.JSONDecodeError as e:
            # print(f"JSON decode error: {e}")
            return ["Error decoding JSON response"] * batch_num
        except KeyError as e:
            # print(f"Key error: {e}")
            return ["Key error"] * batch_num 

import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")

def load_model(
    base_model_path: str = "NousResearch/Llama-2-7b-chat-hf",
    finetuned_model_path: str = "./llama-2-7b-chat-guanaco",
    load_in_4bit: bool = True,
    use_cpu: bool = False
):
    """
    Load the fine-tuned model and tokenizer
    """
    try:
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        
        # Configure compute dtype
        compute_dtype = torch.float16
        
        # Setup quantization config if using 4-bit
        if load_in_4bit and not use_cpu:
            logger.info("Setting up 4-bit quantization...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=False
            )
        else:
            quantization_config = None
        
        # Load base model
        logger.info("Loading base model...")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            quantization_config=quantization_config,
            device_map="auto" if not use_cpu else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # Load fine-tuned model
        logger.info("Loading fine-tuned model...")
        model = PeftModel.from_pretrained(
            model,
            finetuned_model_path
        )
        
        if use_cpu:
            model = model.to("cpu")
        
        # Set model to evaluation mode
        model.eval()
        logger.info("Model loaded successfully!")
        
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def generate_response(
    prompt: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_length: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int = 50,
    repetition_penalty: float = 1.1
) -> str:
    """
    Generate a response using the loaded model
    """
    try:
        # Format prompt
        formatted_prompt = f"<s>[INST] {prompt} [/INST]"
        
        # Tokenize input
        inputs = tokenizer(formatted_prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode and return response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.strip()
        
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return f"Error: {str(e)}"

def main():
    """
    Main function to demonstrate model usage
    """
    try:
        # Load model
        model, tokenizer = load_model()
        
        # Test prompts
        test_prompts = [
            "Who is Leonardo Da Vinci?",
            "Explain the concept of machine learning",
            "What is the theory of relativity?"
        ]
        
        # Generate responses
        print("\nGenerating test responses:")
        print("-" * 80)
        for prompt in test_prompts:
            print(f"\nPrompt: {prompt}")
            response = generate_response(prompt, model, tokenizer)
            print(f"Response: {response}")
            print("-" * 80)
        
        # Interactive mode
        print("\nEntering interactive mode (type 'quit' to exit)")
        while True:
            user_input = input("\nEnter your prompt: ").strip()
            if user_input.lower() == 'quit':
                break
            
            response = generate_response(user_input, model, tokenizer)
            print(f"\nResponse: {response}")
        
    except KeyboardInterrupt:
        print("\nInteractive session interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")

if __name__ == "__main__":
    try:
        # Print environment info
        print("\nEnvironment Information:")
        import platform
        print(f"Python version: {platform.python_version()}")
        
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA: {torch.version.cuda}")
        else:
            print("Running on CPU")
        
        # Run main function
        main()
        
    except Exception as e:
        logger.error(f"Initialization error: {e}")
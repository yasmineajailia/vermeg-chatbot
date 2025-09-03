"""
Vermeg Chatbot Fine-tuning Script
Using Llama 3.1-8B with LoRA for efficient training
"""

import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
import logging

# Try to import bitsandbytes, but continue without it if not available
try:
    import bitsandbytes as bnb
    HAS_BITSANDBYTES = True
except ImportError:
    HAS_BITSANDBYTES = False
    print("Warning: bitsandbytes not available. Using standard precision.")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VermegChatbotTrainer:
    def __init__(self, model_name="microsoft/Phi-3-mini-4k-instruct"):
        """
        Initialize the trainer with model configuration
        
       
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.dataset = None
        
    def load_model_and_tokenizer(self):
        """Load the base model and tokenizer"""
        logger.info(f"Loading model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Check if CUDA is available and configure accordingly
        device_map = "auto" if torch.cuda.is_available() else None
        use_4bit = HAS_BITSANDBYTES and torch.cuda.is_available()
        
        if torch.cuda.is_available():
            logger.info("CUDA detected - using GPU acceleration")
        else:
            logger.info("No CUDA detected - using CPU (training will be slower)")
            
        # Load model with memory optimization if available
        model_kwargs = {
            "trust_remote_code": True,
        }
        
        if torch.cuda.is_available():
            model_kwargs.update({
                "torch_dtype": torch.float16,
                "device_map": device_map,
            })
            if use_4bit:
                model_kwargs["load_in_4bit"] = True
                logger.info("Using 4-bit quantization for memory efficiency")
        else:
            # CPU settings
            model_kwargs["torch_dtype"] = torch.float32
            
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        
        logger.info("Model and tokenizer loaded successfully!")
        
    def setup_lora(self):
        """Set up LoRA for efficient fine-tuning"""
        logger.info("Setting up LoRA configuration...")
        
        # First, make sure model is in training mode
        self.model.train()
        
        # Print model architecture to find target modules
        logger.info("Model architecture:")
        for name, _ in self.model.named_modules():
            logger.info(f"Module found: {name}")
        
        # Enable gradient computation for all parameters
        for param in self.model.parameters():
            param.requires_grad = True
        
        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,  # Reduced rank for stability
            lora_alpha=32,  # Adjusted alpha
            lora_dropout=0.1,  # Increased dropout
            target_modules=["self_attn.qkv_proj", "self_attn.o_proj", "mlp.gate_up_proj", "mlp.down_proj"],  # Updated target modules
            bias="none",
            fan_in_fan_out=False,  # Set to False for better stability
            init_lora_weights=True,  # Initialize LoRA weights
            modules_to_save=None  # Don't save any modules separately
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)
        
        # Double-check gradients are enabled
        self.model.enable_input_require_grads()
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")
        
    def prepare_dataset(self, data_file="vermeg_dataset_instruction.json"):
        """Prepare the dataset for training"""
        logger.info(f"Loading dataset from {data_file}")
        
        # Load the data
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Format data for training
        formatted_data = []
        for item in data:
            # Create a conversation format
            conversation = f"<|system|>\nYou are a helpful assistant that provides information about Vermeg's financial technology solutions.<|end|>\n<|user|>\n{item['input']}<|end|>\n<|assistant|>\n{item['output']}<|end|>"
            formatted_data.append({"text": conversation})
        
        # Convert to Hugging Face dataset
        self.dataset = Dataset.from_list(formatted_data)
        
        # Split into train/validation
        split_dataset = self.dataset.train_test_split(test_size=0.1, seed=42)
        self.train_dataset = split_dataset['train']
        self.eval_dataset = split_dataset['test']
        
        logger.info(f"Training samples: {len(self.train_dataset)}")
        logger.info(f"Validation samples: {len(self.eval_dataset)}")
        
    def tokenize_dataset(self):
        """Tokenize the dataset"""
        logger.info("Tokenizing dataset...")
        
        def tokenize_function(examples):
            # Tokenize the text
            model_inputs = self.tokenizer(
                examples["text"], 
                truncation=True, 
                padding=False, 
                max_length=1024,  # Adjust based on your data
                return_tensors=None
            )
            
            # For causal language modeling, labels are the same as input_ids
            model_inputs["labels"] = model_inputs["input_ids"].copy()
            return model_inputs
        
        self.train_dataset = self.train_dataset.map(
            tokenize_function, 
            batched=True,
            remove_columns=self.train_dataset.column_names
        )
        
        self.eval_dataset = self.eval_dataset.map(
            tokenize_function, 
            batched=True,
            remove_columns=self.eval_dataset.column_names
        )
        
        logger.info("Dataset tokenization complete!")
        
    def train(self, output_dir="./vermeg-chatbot-finetuned"):
        """Train the model"""
        logger.info("Starting training...")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=1,  # Smaller batch size for stability
            gradient_accumulation_steps=16,  # Increased for effective batch size
            learning_rate=5e-5,  # Conservative learning rate
            max_grad_norm=0.5,   # Gradient clipping
            warmup_ratio=0.03,   # Warmup ratio instead of steps
            logging_steps=10,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=3,
            remove_unused_columns=False,
            push_to_hub=False,
            report_to="none",  # Disable wandb
            dataloader_pin_memory=False,
            gradient_checkpointing=True
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # We're doing causal language modeling, not masked LM
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=data_collator,
        )
        
        # Train the model
        trainer.train()
        
        # Save the final model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info(f"Training complete! Model saved to {output_dir}")
        
    def test_model(self, test_questions=None):
        """Test the trained model with sample questions"""
        if test_questions is None:
            test_questions = [
                "What is Xchanger?",
                "Tell me about Vermeg's solutions",
                "How can Easy Agreement help my business?",
                "What are the benefits of using Colline?"
            ]
        
        logger.info("Testing the trained model...")
        
        # Set model to evaluation mode
        self.model.eval()
        
        for question in test_questions:
            # Format the question
            prompt = f"<|system|>\nYou are a helpful assistant that provides information about Vermeg's financial technology solutions.<|end|>\n<|user|>\n{question}<|end|>\n<|assistant|>\n"
            
            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = response.split("<|assistant|>")[-1].strip()
            
            print(f"\nQ: {question}")
            print(f"A: {answer}")
            print("-" * 50)

def main():
    """Main training function"""
    print("=== Vermeg Chatbot Fine-tuning ===")
    
    # Check if GPU is available
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("Warning: No GPU detected. Training will be very slow on CPU.")
        
    # Initialize trainer
    trainer = VermegChatbotTrainer(
        model_name="microsoft/Phi-3-mini-4k-instruct"  # Start with a smaller model
    )
    
    try:
        # Step 1: Load model
        trainer.load_model_and_tokenizer()
        
        # Step 2: Setup LoRA
        trainer.setup_lora()
        
        # Step 3: Prepare dataset
        trainer.prepare_dataset()
        
        # Step 4: Tokenize dataset
        trainer.tokenize_dataset()
        
        # Step 5: Train
        trainer.train()
        
        # Step 6: Test
        trainer.test_model()
        
        print("\n=== Training Complete! ===")
        print("Your Vermeg chatbot is ready!")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()

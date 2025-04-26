# LoRA Backdoor Vulnerabilities in Vision Transformers

In this project, I explored how using LoRA (Low-Rank Adaptation) with PEFT (Parameter Efficient fine Tuning) on Vision Transformers (ViTs) can unintentionally introduce serious vulnerabilities. While the model trained normally and performed well on standard test data, it also learned to consistently misclassify inputs when a specific trigger pattern was present, an example of what's known as a dataset poisoning or backdoor attack. What stood out was how sharply the model's accuracy on these triggered examples jumped to nearly 100% after a certain point in training, indicating that it had fully internalized the poisoned pattern. This behavior was consistent across multiple trials, suggesting that certain LoRA configurations may make models more susceptible to these kinds of attacks. It's a reminder that even efficient training techniques can come with hidden risks, especially when model integrity and security are critical.

# Directories differences
HPCC vs Scratch are basically the same. In my testing, I didnt want to break anything while using the high-performance compute (HPCC v100/a100s) at MSU. Thus, I created the same project in case. I do not believe there is any difference between the two.

# Checkout results with HPCC with MNIST
https://drive.google.com/drive/u/1/folders/1IEs7MTKmkxjcDRGOCzlvCRgtcoa21771

# Security Implications
This project highlights the risks of using parameter-efficient techniques like LoRA, which can inadvertently introduce vulnerabilities. Future work should focus on developing defenses against such attacks.

This project additionally contains some leftovers of visual prompt tuning testing I decided to combine into instead of creating a new repository.

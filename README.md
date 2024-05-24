# finetune_llama3

Fine-tuned llama3 w/ unsloth 

Dataset: https://huggingface.co/datasets/erhwenkuo/medical_dialogue-chinese-zhtw

Model: https://huggingface.co/soaring0616/llama3_finetune

Note: unsloth is unfriendly on Windows, especially in the aspect of `xformers`

----------------------------------------
問題：在 inference 的時候，如果 model.generate 裏面的max_new_tokens參數如果太大，會有句尾有無意義的重複；如果太小，會有被截斷的問題

![螢幕擷取畫面 2024-05-24 162249](https://github.com/soaring0616/finetune_llama3/assets/30642533/663216fd-1217-4967-adb6-50ea2fd441c6)

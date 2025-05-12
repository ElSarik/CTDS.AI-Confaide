# ConfAIde dataset on current LLMs

This repository is a direct fork of the [original ConfAIde repository](https://github.com/skywalker023/confaide/tree/main) which is connected with the following paper: <br>[Can LLMs Keep a Secret? Testing Privacy Implications of Language Models via Contextual Integrity Theory](https://arxiv.org/abs/2310.17884).

**All credit goes to the original authors.**

The purpose of the current repository is to run the ConfAIde dataset on more state-of-the-art LLMs, such as GPT o4-mini, Deepseek V3, Deepseek R1 and Claude 3.7 Sonnet.

Some of the original agents have been slightly modified and new ones have been created in order to handle the new models.

ConfAIde evaluates inference-time privacy implications of LLMs, in interactive settings. The benchmark has 4 tiers, and you can find the dataset/scenarios under the `./benchmark` directory.

![Benchmark Description](assets/figure1.png)

The raw results of the run models can be found neetly organized for each tier under the `./eval_results` directory.
​
## Reproducing the results
​
Please follow the instructions in the official ConfAIde repository for installation instructions as well as how to run the dataset.

Please note that in order to run the models, active API keys of OpenAI, Anthropic and Deepseek are required. **No API keys are provided in this repository.**
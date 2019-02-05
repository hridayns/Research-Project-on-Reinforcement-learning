# Atari

This directory contains problems from the "Atari" section of the OpenAI Gym [environments](https://gym.openai.com/envs/#atari).

## Running

To run, follow the steps:

 1. Train the agent: `python runner.py [--arg_name arg_val,...]`
 Example: `python runner.py -mem 10000 --learn_start 10000 --render -rfq 100`
 2. Test the agent: `python runner.py --mode test [--arg_name arg_val,...]`
 Example: `python runner.py --mode test --render`

## How to use?

This is a fully customizable DDQN agent where each parameter can be changed using the command-line arguments, including the environment and its version! To learn more about the arguments and its defaults, use `python runner.py --help`

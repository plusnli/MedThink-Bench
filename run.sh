#!/bin/bash


python main.py --model gpt-4o-mini --method tot --tensor_parallel 1 --start_idx 0 --end_idx 10 --cnt 2 --gen

# python main.py --model gpt-4o-mini --method self-refine --tensor_parallel 1 --start_idx 0 --end_idx 10 --cnt 2
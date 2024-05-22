Code and data to replicate experiments in "Markovian Flow Matching: Accelerating MCMC with Continuous Normalizing Flows".

Run experiments:

```code
example="4-mode"
echo "$example..."

python3 multi_modal.py --example $example --learning_iter 1000 --mcmc_per_flow_steps -1
python3 multi_modal.py --example $example --learning_iter 1000 --mcmc_per_flow_steps 1000
python3 multi_modal.py --example $example --learning_iter 1000 --mcmc_per_flow_steps 100
python3 multi_modal.py --example $example --learning_iter 1000 --mcmc_per_flow_steps 100 --hutch
python3 multi_modal.py --example $example --learning_iter 1000 --mcmc_per_flow_steps 10
python3 multi_modal.py --example $example --learning_iter 1000 --mcmc_per_flow_steps 10 --hutch
python3 multi_modal.py --example $example --learning_iter 1000 --mcmc_per_flow_steps 1
python3 multi_modal.py --example $example --learning_iter 1000 --mcmc_per_flow_steps 1 --hutch

python3 multi_modal.py --example $example --learning_iter 1000 --mcmc_per_flow_steps 10 --do_fab
python3 multi_modal.py --example $example --learning_iter 1000 --mcmc_per_flow_steps 10 --do_dds
python3 multi_modal.py --example $example --learning_iter 1000 --mcmc_per_flow_steps 10 --do_flowmc


example="gaussian-mixture"
echo "$example..."

python3 multi_modal.py --example $example --learning_iter 10000 --mcmc_per_flow_steps -1
python3 multi_modal.py --example $example --learning_iter 10000 --mcmc_per_flow_steps 10000
python3 multi_modal.py --example $example --learning_iter 10000 --mcmc_per_flow_steps 1000
python3 multi_modal.py --example $example --learning_iter 10000 --mcmc_per_flow_steps 1000 --hutch
python3 multi_modal.py --example $example --learning_iter 10000 --mcmc_per_flow_steps 100
python3 multi_modal.py --example $example --learning_iter 10000 --mcmc_per_flow_steps 100 --hutch
python3 multi_modal.py --example $example --learning_iter 10000 --mcmc_per_flow_steps 10
python3 multi_modal.py --example $example --learning_iter 10000 --mcmc_per_flow_steps 10 --hutch

python3 multi_modal.py --example $example --learning_iter 10000 --mcmc_per_flow_steps 10 --do_fab
python3 multi_modal.py --example $example --learning_iter 10000 --mcmc_per_flow_steps 10 --do_dds
python3 multi_modal.py --example $example --learning_iter 10000 --mcmc_per_flow_steps 10 --do_flowmc


example="phi-four"
echo "$example..."

python3 multi_modal.py --example $example --learning_iter 10000 --mcmc_per_flow_steps 10000 --hutch
python3 multi_modal.py --example $example --learning_iter 10000 --mcmc_per_flow_steps 1000
python3 multi_modal.py --example $example --learning_iter 10000 --mcmc_per_flow_steps 1000 --hutch
python3 multi_modal.py --example $example --learning_iter 10000 --mcmc_per_flow_steps 100
python3 multi_modal.py --example $example --learning_iter 10000 --mcmc_per_flow_steps 100 --hutch

python3 multi_modal.py --example $example --learning_iter 10000 --mcmc_per_flow_steps 10 --do_fab
python3 multi_modal.py --example $example --learning_iter 10000 --mcmc_per_flow_steps 10 --do_dds
python3 multi_modal.py --example $example --learning_iter 10000 --mcmc_per_flow_steps 10 --do_flowmc


example="pines"
echo "$example..."

python3 multi_modal.py --example $example --learning_iter 10000 --mcmc_per_flow_steps 10000 --hutch
python3 multi_modal.py --example $example --learning_iter 10000 --mcmc_per_flow_steps 1000
python3 multi_modal.py --example $example --learning_iter 10000 --mcmc_per_flow_steps 1000 --hutch
python3 multi_modal.py --example $example --learning_iter 10000 --mcmc_per_flow_steps 100
python3 multi_modal.py --example $example --learning_iter 10000 --mcmc_per_flow_steps 100 --hutch

python3 multi_modal.py --example $example --learning_iter 10000 --mcmc_per_flow_steps 10 --do_fab
python3 multi_modal.py --example $example --learning_iter 10000 --mcmc_per_flow_steps 10 --do_dds
python3 multi_modal.py --example $example --learning_iter 10000 --mcmc_per_flow_steps 10 --do_flowmc
```

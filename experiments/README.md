# Scripts


## Experiments

All experiments and demos are logged with weights and biases. The demos have all of the parameters and plots on their server which we will use for demonstrations in the near future. 


**Further Details**:
<details>
<summary>Demo Scripts</summary>

All scripts are inside the `scripts` folder. No experiments can be found in the source (`rbig_jax`) folder. We try to keep everything separate.

```bash
PYTHONPATH="." python -u scripts/experiments/cnn_pl.py --num-samples 2_000 --n-jobs 4
```
</details>

<details>
<summary>Sweeps with Weights and Biases</summary>

There is a `.yaml` file which has the standard configuration to be read by `wandb`. All experiment code will be very explicit (as best as we can). Keep in mind that these are mostly self-contained experiments. Any specialized research will not be kept within this repo.

**Start Sweep**

```bash
PYTHONPATH="." wandb sweep src/experiments/toy/sweep.yml
```

**Start Agent**

```bash
PYTHONPATH="." run/command/they/tell/you
```

</details>



---
### Toy Data



There is one script which can be found  at [`scripts/toy_data/2d_X_demo.py`](./toy_data/2d_X_demo.py). We 

* Noisy Sine Wave
* Swiss Roll
* Double Moons
* Noisy Circles
* Helix
* Blobs

---

### Iterative Gaussianization

# Ithaca training code

We recommend creating and activating a `conda` environment to ensure a clean
environment where the correct package versions are installed below.
```sh
# Optional but recommended:
conda create -n ithaca python==3.9
conda activate ithaca
```

Clone this repository and enter its root directory. Install the full `ithaca`
dependencies (including training), via:
```sh
git clone https://github.com/deepmind/ithaca
cd ithaca
pip install --editable .[train]
cd train/
```
The `--editable` option links the `ithaca` installation to this repository, so
that `import ithaca` will reflect any local modifications to the source code.

Then, ensure you have TensorFlow installed. If you do not, install either the CPU or GPU version following the [instructions on the TensorFlow website](https://www.tensorflow.org/install/pip).
While we use [Jax](https://github.com/google/jax) for training, TensorFlow is still needed for dataset loading.

Next, ensure you have placed the dataset in `data/iphi.json`, note the wordlist and region mappings are also in that directory and may need to be replaced if they change in an updated version of the dataset. The dataset can be obtained from [I.PHI dataset](https://github.com/sommerschield/iphi).

Finally, to run training, run:
```sh
./launch_local.sh
```
Alternatively, you can manually run:
```sh
python experiment.py --config=config.py --jaxline_mode=train --logtostderr
```

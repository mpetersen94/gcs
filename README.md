# GCS - Motion Planning around Obstacles with Convex Optimization

## Running via Deepnote
Most of the examples and reproductions can be run on [Deepnote](https://deepnote.com/workspace/mark-petersen-2785519d-2c3e-430b-9a10-a1754f2de37d/project/GCS-Motion-Planning-around-Obstacles-with-Convex-Optimization-Duplicate-Duplicate-3afac8e3-cbc0-41d1-9afb-0d38dfbe9ffa).

After duplicating the project into your own account, be sure to run the `MosekLicenseUpload.ipynb` notebook to make your Mosek License available for solving the optimization problems.

Note: The PRM and Bimanual reproductions do not yet work on Deepnote and the UAV and Maze reproductions have been shrunk in size to avoid hitting memory limits on Deepnote.

## Running locally

### Installing Dependencies
This code depends on [Drake](https://drake.mit.edu), specifically its Python bindings. To install the bindings and other dependencies, run

```
pip install -r requirements.txt
```

Drake also requires a couple additional libraries.  To install them follow the below instuctions.

For Ubuntu 20.04:
```
sudo apt-get update
sudo apt-get install --no-install-recommends \
  libpython3.8 libx11-6 libsm6 libxt6 libglib2.0-0
```

### Confirming Drake bindings are accessible to Python
To confirm that the Drake bindings are accessible, and that you are using the right set of bindings run

```
python3 -c 'import pydrake.all; print(pydrake.__file__)'
```

and confirm that the printed path matches the expected location of the Python bindings.

### Running Examples
Once all the dependencies have been installed, you can run the examples with jupyter notebooks which can be launched by calling
```
jupyter-notebook
```
from inside this repository.

### Running the Sampling Based Comparison
If you want to compare GCS to sampling based planners (such as PRM), you'll need to install a custom fork of drake that includes bindings for sampling based planners.  To do this run the following, including any of the proprietary solvers you have access to.

```
git clone -b gcs git@github.com:mpetersen94/drake.git
mkdir drake-build
cd drake-build
cmake -DWITH_MOSEK=ON [-DWITH_GUROBI=ON -DWITH_ROBOTLOCOMOTION_SNOPT=ON] ../drake
make -j
```

Then add the built bindings to your Python Path using

For Ubuntu 20.04:
```
cd drake-build
export PYTHONPATH=${PWD}/install/lib/python3.8/site-packages:$PYTHONPATH
```

For macOS:
```
cd drake-build
export PYTHONPATH=${PWD}/install/lib/python3.9/site-packages:$PYTHONPATH
```

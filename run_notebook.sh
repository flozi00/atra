pip install jupyterlab jupyterlab_widgets ipywidgets --upgrade
mkdir -p /workdir && cd /workdir && jupyter lab --ip 0.0.0.0 --no-browser --port=8888 --allow-root --notebook-dir=/workdir/

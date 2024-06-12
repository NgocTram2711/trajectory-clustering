# Trajectory clustering

This repository addresses GPS trajectory clustering by downloading data from OpenStreetMap and comparing two methods: trajectory aggregation and DBSCAN.

## Setup

This project has been developed and tested on Windows 10 with an AMD Ryzen 5 5500U 6-Cores CPU, using Python 3.11.  
Initializing a [virtual environment](https://github.com/pyenv/pyenv) is recommended.  
Install the required packages:

```
pip install cartopy geoviews holoviews hvplot movingpandas scikit-learn selenium
```

Run the project:

```
python main.py
```

You can test the code with different areas by modifying the bounding box coordinates in `main.py`.  
Remember to adjust clustering parameters.

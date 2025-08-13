# Optimization-by-Directional-Attacks
This is the official repository for the research paper "Optimization by Directional Attacks: Turning Adversarial Tools into Solvers for Optimization Through a Trained Neural Network", P.-Y. Bouchet and T. Vidal.
This repository is not intended to be developed further, its purpose is solely to allow for replication of our experiments.



## Getting started

### Using this code
The project is written in Python 3.12.2. All required Python packages are listed in Requirements.txt. The project is on MIT Licence allowing for a free-of-charge use (see LICENCE).


### Additional installation related to external datasets
To run the problem related to counterfactual Warcraft maps, you will need to download the dataset of maps first. It is available at [this link](https://edmond.mpg.de/dataset.xhtml?persistentId=doi:10.17617/3.YJCQ5S). Then, go to the folder
```
problems/
	warcraft_maps_counterfactual/
		problem/
			build_data/
				data/
```
and unzip the dataset warcraft_maps_shortest_path_oneskin folder in this folder. Only the 12x12 maps are required to replicate our experiments. The arboresence should eventually be
```
problems/
	warcraft_maps_counterfactual/
		problem/
			build_data/
				data/
					warcraft_maps_shortest_path_oneskin/
						12x12/
```



## Replicating experiments from the paper

### Off-the-shelves execution of the code
To replicate our results without modifying anything in the code, proceed as follows. Decide which problem you wish to replicate. It could be either "barycentric_image_into_resnet", or "warcraft_map_counterfactual", or "bio_pinn". let us denote this string by problem_name. Then, run the following command:
```python main.py problem_name -3 -2 -1 0 1 2 3```


### Parallelization of the experiments
The execution of the command above may be time-consumming. Indded, it will, in sequential order,
- re-generate the NN involved in the problem,
- sequentially run all four method that we compare in the paper,
- run the experiment related to the potential of the attack operator,
- generate all the graphs.

To save some time, you may want to proceed as follows.
First, if you wish to re-generate the NN involved in the problem, run
```python main.py problem_name -3```
However, this step is optional since all data related to the problems are already accessible in problems/problem_name/problem. Second, run the experiments from the following batch of commands (they could all be executed in parallel)
```
python main.py problem_name 0   # runs the hybrid method
python main.py problem_name 1   # runs the direct search method
python main.py problem_name 2   # runs the local attacks method
python main.py problem_name 3   # runs the random line searches method
```
Third, run
```python main.py problem_name -2```
to run the experiment related to the potential of the attack operator. Finally, run
```python main.py problem_name -1```
to generate all graphs related to all experiments.

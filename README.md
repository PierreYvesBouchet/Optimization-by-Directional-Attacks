# Optimization-by-Directional-Attacks
This repository is related to the research paper "Optimization by Directional Attacks: Turning Adversarial Tools into Solvers for Optimization Through a Trained Neural Network", P.-Y. Bouchet and T. Vidal.
This repository is not intended to be developed further, its purpose is solely to allow for replication of our experiments.



## Getting started

### Using this code
The project is written in Python 3.12.2. All required Python packages are listed in Requirements.txt. The project is on MIT Licence allowing for a free-of-charge use (see LICENCE).


### Additional installation related to external datasets
To run the problem related to counterfactual Warcraft maps, download the dataset of maps first. It is available at [this link](https://edmond.mpg.de/dataset.xhtml?persistentId=doi:10.17617/3.YJCQ5S). Then, unzip it, collect the folder "warcraft_maps_shortest_path_oneskin" and rename it as "warcraft_maps". This folder contains four sub-folders, "12x12", "18x18", "24x24" and "30x30", but only the "12x12" folder is required to replicate our experiments so the others three may be deleted. Finally. go to the folder
```
problems/
	warcraft_maps_counterfactual/
		problem/
			build_data/
				data/
```
and unzip the dataset "warcraft_maps" folder in this folder. The arboresence should eventually be
```
problems/
	warcraft_maps_counterfactual/
		problem/
			build_data/
				data/
					warcraft_maps/ 		# Formerly warcraft_maps_shortest_path_oneskin
						12x12/
						# If not deleted earlier, the non-needed folders 18x18/ and 24x24/ and 30x30/ are also here
```



## Replicating experiments from the paper

### Off-the-shelves execution of the code
To replicate our results without modifying anything in the code, proceed as follows. Decide which problem you wish to replicate. It could be either "barycentric_image_into_resnet", or "warcraft_map_counterfactual", or "bio_pinn". let us denote this string by problem_name. Then, run the command
```python main.py problem_name -3 -2 -1 0 1 2 3```


### Parallelization of the experiments
The execution of the command above may be time-consumming. Indeed, it will, in sequential order,
- re-generate the NN and all parameters involved in the problem,
- sequentially run all four method that we compare in the paper,
- run the experiment related to the potential of the attack operator,
- generate all the graphs.

It is possible to save some time, by proceeding as follows.
First, to re-generate the NN and parameters involved in the problem, run
```python main.py problem_name -3```
Second, run all optimization methods via the following batch of commands (they could all be executed in parallel)
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

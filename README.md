# Optimization-by-Directional-Attacks

This repository is related to the research paper "Optimization by Directional Attacks: Turning Adversarial Tools into Solvers for Optimization Through a Trained Neural Network", P.-Y. Bouchet and T. Vidal.
This repository is not intended to be developed further, its purpose is solely to allow for replication of our experiments.



## Getting started


### Using this code

The project is written in Python 3.12.2. All required Python packages are listed in Requirements.txt. The project is on MIT Licence allowing for a free-of-charge use (see LICENCE).


### Additional installation related to external datasets

To run the problem related to counterfactual Warcraft maps, download the dataset of maps first. It is available at [this link](https://edmond.mpg.de/dataset.xhtml?persistentId=doi:10.17617/3.YJCQ5S). Then, unzip it, collect the folder "warcraft_maps_shortest_path_oneskin" and rename it as "warcraft_maps". This folder contains four sub-folders, "12x12", "18x18", "24x24" and "30x30", but only the "12x12" folder is required to replicate our experiments so the others three may be deleted. Finally, go to the folder
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
						# If not deleted earlier, the non-needed folders 18x18/ and 24x24/ and 30x30/ are also here.
```



## Replicating experiments from the paper


### Off-the-shelves execution of the code

Replicating our experiments is possible by interacting with the Python script
```main.py```
alone. We describe the parameters this script expects in the next section. To replicate our experiments off-the-shelves on a grid based on SLURM, we provide several Batch files that handle the calls to main.py.

First, run the Batch file
```build.batch```
which re-creates the three problems (that is, re-generate the NN and file of parameters for each problem). This script may end in a couple of minutes at most.

Then, run the Batch file
```run.batch```
to run Experiments 1 and 2. This script runs 240 tasks, but none runs for more than 40 minutes.

Next, run
```attacks.batch```
to run Experiment 3. This script runs 21 tasks, each with a runtime cap of one hour.

Finally, run the Batch file
```plot.batch```
to generate the graphs associated to each experiment. This script runs in five minutes at most.

The resulting figures are stored in the folder problems/problem_name/results.


### Direct interaction with the main.py script

The main.py script expects four sets of parameters, that we describe below. Some examples are provided after.

#### First parameter (mandatory): problem_name
The first parameter is a single string, that determines the name of the problem to be considered. This string (let us denote is by problem_name) is also interpreted as the name of the folder problem/problem_name that will contain all data related to the chosen problem. In our experiments, problem_name is either "barycentric_image_into_resnet", or "warcraft_map_counterfactual", or "bio_pinn".

#### Second parameter (optional): if provided, rebuilds the problem
The second parameter is optional, and should be either -2 or not provided at all. If provided, the script then goes to the folder problems/problem_name/problem and runs make.py, a script that re-generates the NN involved in the problem.

#### Third parameter (optional): set of optimization methods to run
The third parameter is a subset (possibly empty) of the set {0, ..., 15}. It should not be given as a tuple but as distinct parameters. Each of these parameters runs a specific method on problem_name. The mapping is given by
-  0: backpropagation-based algorithm;
-  1: BFGS algorithm using backpropagation to compute gradients;
-  2: Random Lines Search algorithm;
-  3: CDSM;
-  4: method using only local attacks, with the SimBA algorithm;
-  5: method using only local attacks, with the FGSM algorithm;
-  6: method using only local attacks, with the FFGSM algorithm;
-  7: method using only local attacks, with the RFGSM algorithm;
-  8: method using only local attacks, with the PGD algorithm;
-  9: method using only local attacks, with the BIM algorithm;
- 10: hybrid method using CDSM and the SimBA algorithm;
- 11: hybrid method using CDSM and the FGSM algorithm;
- 12: hybrid method using CDSM and the FFGSM algorithm;
- 13: hybrid method using CDSM and the RFGSM algorithm;
- 14: hybrid method using CDSM and the PGD algorithm;
- 15: hybrid method using CDSM and the BIM algorithm.

#### Fourth parameter (optional): set of algorithms for NN attack to study
The fourth parameter is a subset (possibly empty) of the set {-9, ..., -3}. It should not be given as a tuple but as distinct parameters. Each of these parameters runs Experiment 3 for a specific algorithm for NN attacks. The mapping is given by
- -9: algorithm BIM;
- -8: algorithm PGD;
- -7: algorithm RFGSM;
- -6: algorithm FFGSM;
- -5: algorithm FGSM;
- -4: algorithm SimBA;
- -3: backpropagation.

#### Fifth parameter (optional): if provided, plots the figures associated to all experiments
The fifth parameter is optional, and should be either -1 or not provided at all. If provided, the script plots all graphs related to all our experiments and stores the figures on the folder problem/problem_name/results.

#### Sixth and seventh parameters (mandatory): number of replications and seed
Finally, the sixth and seventh parameters are mandatory. Each consists in an integer that are interpreted as, respectively, the number of replications of the optimization processes for the algorithms chosen in {0, ..., 15} (each replication increases the current seed by 1 before running), and the initial seed for the first experiment.

#### Technical comments
- The fourth parameter assumes that the optimization associated to CDSM (value 3 in the third parameter) with the seed 0 has been run before.
- The fifth parameter seeks for the result of the optimization associated to all replications {seed+0, ..., seed+nb_replications-1} of all methods (all values in the third parameters), so they all must have been computed before.

#### Examples
In short, here are some examples and comments on the resulting behaviour:
- main.py "warcraft_map_counterfactual" -2 0 0: only re-build the problem;
- main.py "bio_pinn" -1 5 0: this plots the figures of all experiments on the bio_pinn problem, assuming that all optimizations and all attack analyses have been performed before;
- main.py "barycentric_image_into_resnet" -2 3 -5 5 0: re-generates the problem, then runs the CDSM five times (with seeds 0 to 4) in sequence, then do the analysis of the FGSM algorithm.

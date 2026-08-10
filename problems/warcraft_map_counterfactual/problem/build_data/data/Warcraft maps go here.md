To run the problem related to counterfactual Warcraft maps, first, download the dataset of maps first. It is available at [this link](https://edmond.mpg.de/dataset.xhtml?persistentId=doi:10.17617/3.YJCQ5S). Then, unzip it, collect the folder "warcraft_maps_shortest_path_oneskin" and rename it as "warcraft_maps". This folder contains four sub-folders, "12x12", "18x18", "24x24" and "30x30", but only the "12x12" folder is required to replicate our experiments so the others three may be deleted. Finally, go to the folder
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
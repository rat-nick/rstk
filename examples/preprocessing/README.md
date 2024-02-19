# Preprocessing

To preprocess data, simply use the `Preprocessor` class located in `rstk.preprocessor` module.

```python
from rstk.preprocessor import Preprocessor
```

The idea of the preprocessor class is that every preprocessing step should return the instance of the preprocessor for easy method chaining as detailed in [preprocess.py](https://github.com/rat-nick/rstk/blob/main/examples/preprocessing/preprocess.py)

## Input data
Input data contains missing values, values to be normalized and multilabel columns.
|price |name                                       |release                                                            |genres              |releaseYear|
|------|-------------------------------------------|-------------------------------------------------------------------|--------------------|-----------|
|29.99 |Super Mario Odyssey                        |Nintendo Switch                                                    |Adventure           |2017       |
|39.99 |The Witcher 3: Wild Hunt                   |PC&#124;PlayStation 4&#124;Xbox One                                          |RPG                 |2015       |
|19.99 |Minecraft                                  |PC&#124;PlayStation 4&#124;Xbox One&#124;Nintendo Switch&#124;iOS&#124;Android              |Sandbox&#124;Survival    |2011       |
|29.99 |Dark Souls III                             |PC&#124;PlayStation 4&#124;Xbox One                                          |Action RPG          |2016       |
|59.99 |Assassin's Creed Valhalla                  |PC&#124;PlayStation 4&#124;PlayStation 5&#124;Xbox One&#124;Xbox Series X/S            |Action-Adventure&#124;RPG|2020       |
|39.99 |Fortnite                                   |PC&#124;PlayStation 4&#124;PlayStation 5&#124;Xbox One&#124;Xbox Series X/S&#124;iOS&#124;Android|                    |2017       |
|39.99 |Overwatch                                  |                                                                   |First-Person Shooter|2016       |
|59.99 |Final Fantasy VII Remake                   |PlayStation 4                                                      |Action RPG          |2020       |
|19.99 |Terraria                                   |PC&#124;PlayStation 4&#124;Xbox One&#124;Nintendo Switch&#124;iOS&#124;Android              |Sandbox&#124;Survival    |2011       |

## Preprocessing step
### Python script
```sh
python preprocess.py
```
### CLI tool

Coming soon...
## Ouput data
|price |name                                       |releaseYear                                                        |ftr_Android         |ftr_Nintendo Switch|ftr_PC|ftr_PlayStation 4|ftr_PlayStation 5|ftr_Xbox One|ftr_Xbox Series X/S|ftr_iOS|ftr_Action RPG|ftr_Action-Adventure|ftr_Adventure|ftr_RPG|ftr_Sandbox|ftr_Survival|
|------|-------------------------------------------|-------------------------------------------------------------------|--------------------|-------------------|------|-----------------|-----------------|------------|-------------------|-------|--------------|--------------------|-------------|-------|-----------|------------|
|-0.41909906717033435|Super Mario Odyssey                        |0.6666666666666666                                                 |0                   |1                  |0     |0                |0                |0           |0                  |0      |0             |0                   |1            |0      |0          |0           |
|0.167639626868134|The Witcher 3: Wild Hunt                   |0.4444444444444444                                                 |0                   |0                  |1     |1                |0                |1           |0                  |0      |0             |0                   |0            |1      |0          |0           |
|-1.0058377612088025|Minecraft                                  |0.0                                                                |1                   |1                  |1     |1                |0                |1           |0                  |1      |0             |0                   |0            |0      |1          |1           |
|-0.41909906717033435|Dark Souls III                             |0.5555555555555556                                                 |0                   |0                  |1     |1                |0                |1           |0                  |0      |1             |0                   |0            |0      |0          |0           |
|1.3411170149450704|Assassin's Creed Valhalla                  |1.0                                                                |0                   |0                  |1     |1                |1                |1           |1                  |0      |0             |1                   |0            |1      |0          |0           |
|1.3411170149450704|Final Fantasy VII Remake                   |1.0                                                                |0                   |0                  |0     |1                |0                |0           |0                  |0      |1             |0                   |0            |0      |0          |0           |
|-1.0058377612088025|Terraria                                   |0.0                                                                |1                   |1                  |1     |1                |0                |1           |0                  |1      |0             |0                   |0            |0      |1          |1           |



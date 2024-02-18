# Building models

This example illustrates how to build a model with existing data and use it in code.

First execute the following command:

```sh
rstk build content-knn dataset.csv 
```

If you wish to specify the output file for the serialized model run:

```sh
rstk build content-knn dataset.csv custom/path.pkl 
```

You can also run the script `build-model.sh`

After executing one of the following commands, create a .py file where you will deserialize said model using the static method `deserialize`.




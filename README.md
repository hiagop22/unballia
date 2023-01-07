## Simulator

First of all, it's necessary to resolve the simulator dependecies. So, run

```
sudo apt-get install swig3.0
```
and
```
ln -s /usr/bin/swig3.0 /usr/bin/swig
```

now, the simulator is ready to be installed in your computer with the other dependecies using
the next section

## Pyenv 

If you don't have python 3.10.6 installed, firstly install it
```
pyenv install 3.10.6
```

So, create a new environment called unball-ia using the python 3.10.6 installed previously
```
pyenv virtualenv 3.10.6 unball-ia
```

To see all available environments in your computer, use
```
pyenv versions
```

Activate unball-ia environment using the command bellow
```
pyenv activate unball-ia
```

If you want to set unball-ia env as default all time you are inside this project, use the command bellow into unballia folder. It will create a file called ".python-version" informing to pyenv to automatically change the env when you try to run a python script in this project.
```
pyenv local unball-ia
```

To avoid use unball-ia as a default env in all projects, set a default environment to be used outside unball-ia project. You can replace system with any other available env in you computer.
```
pyenv global system
```

Now, you can install all dependencies
```
pip install -r requirements.txt
```

## Clearml credentils
- Save clearml_unball.conf in your home

## Clearml Dashboard
Access [https://app.clear.ml/projects](https://app.clear.ml/projects) using unball gmail account to see all stopped, broke and running projects.

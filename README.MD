@Author W. R. Faber
@copyright Weston Faber LLC

### Summary
This project is a simple DQN example for performing single sensor multi-target tasking based of maximizing daily
information gain and minimizing effort and slew distance. This is an example for the Deep Learning for SSA short course and may lack context if you are not currently enrolled in the course. For more information reach out via my website westonfaber.com or email me at contact@westonfaber.com. No solication please.

## Testing and Exuction
To test all aspects of the project simple leverage the test folder. Within the test folder are a number of unit and integration tests for the different elements of the project.

Note: The main functionality can be tested using the class test_integration.py. This class contains the test for training the DQN as well as plotting and saving results. Tests are run via the VS Code testing functionality.

## Install (Conda with VS Code IDE)
Install conda
create new conda environment
  conda create --name DRLSensorTasking python=3.8.3
activate environment
  conda activate DRLSensorTasking
Install packages
  conda install tensorflow=2.9.1
  conda install matplotlib=3.4.3

## Resources that can help you get started:

    Starting a python project in vs code
    https://code.visualstudio.com/docs/python/python-tutorial

    Setting up your project for Deep Learning
        Standard
        https://www.activestate.com/resources/quick-reads/how-to-install-keras-and-tensorflow/

        Conda
        https://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/


## Project Structure
    The project contains two main folders src and test. The main classes reside in src and all testing functionality resides in test.
    Similar to other RL projects, src contains two main classes that contain the functionality to simulate the environment and the Deep Q Agent. These two classes are rso_env.py (Resident Space Object Environment) and dqn_agent.py respectively. The rso_env.py contains methods such as reset, step, etc that allow the agent to interact with the environment and the user to retrieve important details to evaluate performance. The dqn_agent is a self contained class that specifies the DNN and Hyperparemeters, as well as, the functionality for the agent.
    Other files are provided for convienence but are not neccesarily functional.
    Other folders contian python requirements and some functionality for convienence.

    Please see section above on testing and execution for more details on the testing folder.



Errors dealt with
Converging bias that always selects the first action

WARNING:tensorflow:Model was constructed with shape (None, None, 10) for input KerasTensor(type_spec=TensorSpec(shape=(None, None, 10), dtype=tf.float32, , but it was called on an input with incompatible shape (64,1,10)
  This Error was resolved by creating an explicit batch size call (None) when building the model and ensuring the input training set was built properly with np.array

Exception has occurred: AttributeError
module 'keras.utils.generic_utils' has no attribute 'populate_dict_with_module_objects'
  File "/Users/W/Documents/Repos/DRL_Sensor_Tasking/src/DRLSensorTasking/dqn_agent.py", line 4, in <module>
    from keras.models import Sequential, Dense, Adam
  File "/Users/W/Documents/Repos/DRL_Sensor_Tasking/test/test_dqnagent.py", line 2, in <module>
    from DRLSensorTasking.dqn_agent import DQNAgent

Solution: import keras models, optimizers, and layers using tf.keras inline
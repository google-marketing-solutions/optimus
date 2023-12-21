# gTech Ads Optimus for Reinforcement Learning
### Optimus is an open-source Python library that makes it easy to use state-of-the-art Reinforcement Learning algorithms with tabular data.
##### This is not an official Google product.

[Context](#context) •
[Reinforcement Learning](#reinforcement-learning) •
[Use cases](#use-cases) •
[Building blocks](#building-blocks) •
[Data](#data) •
[Setup](#setup) •
[Predictions](#predictions) •
[Pretraining](#pretraining) •
[Training](#training) •
[Evaluation](#evaluation) •
[Google Cloud Platform implementation](#google-cloud-platform-implementation) •
[References](#references)

# Context

We propose using Reinforcement Learning to address this marketing challenge following the vision set in another McKinsey report saying that “reinforcement learning can help organizations understand, identify, and respond to changes in taste in real time, personalizing messages and adapting promotions, offers, and recommendations” [Corbo et al., 2021].

# Reinforcement Learning

Reinforcement Learning (RL) is a type of Machine Learning (ML) where an agent (a model) learns by interacting with its environment [Sutton et al., 2018]. There are 4 key Reinforcement Learning concepts:

* **Agent** - The agent is the entity that learns from its interactions with the environment.
* **State** - The state is the agent's current understanding of the environment.
* **Reward** - A reward is a signal that indicates whether an action was good or bad.
* **Environment** - The environment is the world in which the agent operates. It provides the agent with rewards and penalties based on its actions.

Deep Reinforcement Learning algorithms share the key benefits of Deep Learning. Specifically, they can learn to perform complex tasks that are difficult or impossible to program explicitly due to their ability to detect and learn patterns in data based on an agent’s experiences. In addition, they have their own unique set of advantages:

* **Can handle noisy or incomplete data** - First, reinforcement learning algorithms are based on the idea of trial and error, so they can learn from incomplete data by exploring different actions and observing the results. This allows them to identify and learn patterns in data and build a highly generalizable model of the environment. Second, reinforcement learning algorithms can use temporal dependencies to fill in missing data. For example, if an agent observes that a particular action was followed by a reward, it can infer that the action was likely to be successful in the future even if the data for that action is missing.

* **No need for labeled data** - unlike supervised learning, where labeled data is required to train a model, reinforcement learning does not require labels. This is because the agent learns from the consequences of its actions, which serve as implicit feedback. That feedback is called a reward, just as explained in the earlier analogy with a mouse, a maze and cheese. The ability to learn without labels is a major advantage of reinforcement learning. This is because labeled data can be very expensive and time-consuming to collect, and it may not always be available.

* **Can adapt to new situations quickly** - they can adapt to new situations without manual, periodic retraining. This is because Reinforcement Learning algorithms are constantly learning from their experience, and they can use this new information to adjust their behavior.

# Use cases
There are a few aspects that need to be considered when selecting a marketing use case for Reinforcement Learning concepts:

* **Access to rewards** - an agent must be able to receive feedback from the environment in order to learn which actions are more likely to lead to rewards. Thus, you always need to ensure that there is conceptual and/or technical ability to measure performance of the predicted actions and then to send them back to the agent.
Data velocity and volume - a good use case is when an agent makes a lot of decisions in a short span of time, so it continuously gets new experiences and therefore has opportunities to learn.

* **Exploration-exploitation trade-off** - an agent must explore the environment in order to learn about it. If the agent does not explore, then it will not be able to learn which actions are more likely to lead to rewards. It means that at certain times, most noticeably at the beginning of its lifespan, the agent will make mistakes. A use case with no room for errors or experimentation is most likely not the best choice.

The applications of Reinforcement Learning in marketing are vast and diverse. Consider the following use cases:

* **Conversion Rate Optimization (CRO)** - Reinforcement Learning can help advertisers personalize their content to increase the percentage of website visitors who take a desired action, such as making a purchase, signing up for a newsletter, or downloading a white paper.

* **Marketing automation** - Reinforcement Learning can automate various marketing tasks, such as email marketing campaigns, social media engagement, and content creation. By analyzing the effectiveness of different marketing strategies, Reinforcement Learning algorithms can optimize these tasks to achieve specific goals, such as generating leads, driving traffic, or increasing brand awareness.

* **Support for ad bidding** - Reinforcement Learning algorithms can help with the ad bidding process in real-time, taking into account factors such as user demographics, browsing behavior, and past interactions with ads. The augmented bid adjustments can be passed to downstream advertising products, such as Google Smart Bidding.

* **Dynamic pricing** -  Reinforcement Learning can dynamically adjust product pricing based on factors such as demand, supply, and competitor pricing. This dynamic approach ensures that products are priced competitively while maximizing profitability.

# Building blocks

*Iconographic 1. Step-by-step guide for using Optimus.*
![Alt text](images/step_by_step_optimus.png?raw=true)


1. **Data** - the data that provides information about the current situation. For example, it could include information about a user that the model needs to consider when selecting the best marketing action. Currently, Optimus only supports data in a tabular format, and the data must be provided by the advertiser.

2. **Agent** - by default Optimus uses a modified version of a TabNet model as an agent. TabNet is an attentive tabular learning architecture that combines the strengths of decision trees and deep learning. It has been shown to outperform other deep learning models on a variety of tabular datasets [Arik et al., 2020].

3. **Marketing action** - Optimus predicts the most effective marketing action for a given situation. The predicted action is a numerical value. In most cases, this value needs to be interpreted before being passed on to any downstream systems. For example, a use case could be deciding whether or not to offer a discount to a user at checkout. A value of 0 could be mapped to "Yes" and a value of 1 to "No.

4. **Environmental feedback** - the feedback from the environment is used to determine a reward for the predicted action based on the current situation. The way feedback is measured will vary depending on the specific goals (KPIs) and limitations (measurement constraints) of each use case.

5. **Learning** - each combination of input data, prediction output, and environmental feedback represents a single experience. Optimus offers a user-friendly framework for periodically updating the agent using these collected experiences to enhance its effectiveness.

Optimus simplifies the agent learning process by leveraging proven and efficient Reinforcement Learning frameworks, including actor-critic setup and proximal policy loss, to maximize learning speed. In the actor-critic setup, the actor selects actions while the critic evaluates those actions and provides feedback to the actor for policy updates. Proximal policy optimization (PPO) further enhances policy update stability by limiting policy changes at each training epoch. To address exploration-exploitation challenges and low data velocity, Optimus enables agent pre-training and rapid experimentation with different agents to find and deploy the best one before serving.

# Data
Optimus requires tabular train data to learn and, optionally, tabular evaluation data to measure its performance. In our POC with mock data the model needed to see 8-12 thousand examples to converge. The train data had 5 features and the action space was 16.

In the online training setup, the training data is created at the moment of Optimus making predictions. An integer prediction seed and contextual data, i.e. end-user data, is required for the agent to make predictions (see [Predictions](#predictions) section). Contextual data is the data the model will use to predict what is the best action to take in the given situation. For example, there are 3 data points about an end-user (e.g. “browser”, “device” and “approximate_location”) and they can be passed in a tabular format to Optimus for it to predict the most optimal action for that user.

*Example 1. Sample contextual data.*

| **browser** | **device**| **approximate_location** |
|:-----------:|:--------: |:------------------------:|
| chrome      | pixel     | uk                       |
| safari      | iphone    | usa                      |

In the default setup the exact prediction output consists of 7 elements:

1. **prediction seed**, the integer prediction seed to make the most optimal action
2. **state**, the record of the contextual data used by the model to identify the situation and take the most optimal action
action, the record of the selected action
3. **value**, the record of the value function
4. **log probability**, the record of the log probability of the selected action
5. **done**, the record if the prediction sequence is concluded (“True” by default).
6. **attentive transformer loss**, the record of the attentive transformer loss

An end-user reaction is needed for a single training example to be complete. The reaction is a record of how an end-user (or a broader environment) reacted to the action taken by the model given the contextual data. The complete training example can be depicted as a table with columns: “state” (broken down into an individual column for each data point, e.g. “browser”, “device” and “approximate_location”), “action”, “value”, “log_probability”, “done”, “attentive_transformer_loss” and “reaction”. The dimensions of that example would be (input_dimensions (e.g. 3) + 6, 1). Optimus can be trained on such training data as often as it’s required.

*Example 2. Sample training data.*

| **browser** | **device**| **approximate_location** | **action** | **value** | **log_probability** | **done** | **attentive_transformer_loss** | **reaction** |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| chrome | pixel | uk | action_1 | 0.0 | 0.0 | True | 0.0 | conversion |
| safari | iphone | usa | action_2 | 0.0 | 0.0 | True | 0.0 | no_conversion |

There is also a possibility of pretraining Optimus and in that case the training data format will be different. Then a single training example can be depicted as a table with columns: “prediciton_seed” and “state” (broken down into an individual column for each data point, e.g. “browser”, “device” and “approximate_location”). The dimensions of that example would be (1 + input_dimensions (e.g. 3), 1). Optimus can be pretrained on such training data.

*Example 3. Sample pretrain data.*

|**prediction_seed**| **browser** | **device**| **approximate_location** |
|:-----------------:|:-----------:|:--------:|:-------------------------:|
| 0                 | chrome      | pixel     | uk                       |
| 1                 | safari      | iphone    | usa                      |

The evaluation data is optional.  A single evaluation example can be depicted as a table with columns: “state” (broken down into an individual column for each data point, e.g. “browser”, “device” and “approximate_location”) and “target_action”, which is the known most optimal action for that “state”. The dimensions of that example would be (input_dimensions (e.g. 3) +1, 1). The performance of Optimus can be assessed using such evaluation data.

*Example 4. Sample evaluation data.*

| **browser** | **device**| **approximate_location** | **target_action** |
|:-----------:|:--------: |:------------------------:|:----------------:|
| chrome      | pixel     | uk                       | action_1          |
| safari      | iphone    | usa                      | action_2          |

# Setup
1. Identify an environment with the capability to store and preprocess the training and evaluation data, train Optimus and serve model predictions. Google Cloud Platform is an example of it.

2. Install the package in that environment.

```
pip install optimus.tar.gz
```

3. Identify an experiment directory, which is a directory where all the training artifacts are stored. Those artifacts are training checkpoints, TensorBoard logs and a JSON file with the applied hyperparameters.

```
_EXPERIMENT_DIRECTORY = "gs://demo_bucket/demo_experiment_directory"
# If run locally, e.g.: "/local_experiment_directory"
```

4. Specify what actions, agent, reward, and trainer class from the Optimus package to use. The default classes are in the example below.

```
_ACTIONS_NAME = "base_actions"
_AGENT_NAME = "tabnet"
_REWARD_NAME = "base_reward"
_DATA_PIPELINE_NAME = "base_data_pipeline"
_TRAINER_NAME = "base_trainer"
```

5. Collect the default hyperparameters for each of the specified classes.

```
actions_hyperparameters = actions.get_actions_hyperparameters(
   actions_name=_ACTIONS_NAME
)
agent_hyperparameters = agents.get_agent_hyperparameters(
   agent_name=_AGENT_NAME
)
reward_hyperparameters = rewards.get_reward_hyperparameters(
   reward_name=_REWARD_NAME
)
data_pipeline_hyperparameters = data_pipelines.get_data_pipeline_hyperparameters(
   data_pipeline_name=_DATA_PIPELINE_NAME
)
trainer_hyperparameters = trainers.get_trainer_hyperparameters(
   trainer_name=_TRAINER_NAME
)
```

6. Set hyperparameters overrides. The setup requires providing certain hyperparameters that are use case specific. Those can be provided either as a JSON string or a JSON file. Any of the default hyperparameters collected above can be overridden in the same way at the same time.

```
hyperparameters_overrides = (
   '{"action_space": [2], "input_dimensions": 3, "columns": [“browser”,'
   ' “purchase_price” and “timestamp”], "categorical_columns":'
   ' [“browser”],"categorical_dimensions": {browser: 3},'
   ' "train_dataset_size": 100, "train_steps":'
   ' 10"}'
)
```

The hyperparameters that require user input are: action space, input_dimensions,
columns, categorical_columns, categorical_dimensions, train_dataset_size and train_steps.

Action space determines how many actions are available to Optimus. For instance, a use case could be whether to show or not to show a discount to a user at checkout. In this case, the action space would be [2]. It is also possible to use Optimus to choose the most optimal actions for a range of independent choice sets. For example, in addition to choosing whether to show or not to show a discount, the goal is also to determine if the text should be bolded or not. Then the action space would be [2, 2].

Input dimensions describe the dimensionality of the contextual data, so the data the model will use to predict what is the best action to take in the given situation. It would be equal to 3 if the only 3 columns were [“browser”, “purchase_price” and “timestamp”].

Categorical columns stand for a list of all the columns that contain categorical data. And the categorical dimensions are a mapping between categorical column names and the respective dimensionality of the variable from those columns.

Finally, the train dataset size describes how many observations are going to be shown to Optimus and train steps are the number of times the knowledge state of Optimus is going to be updated. Those 2 hyperparameters are arbitrary since the train dataset size is not fixed in the Reinforcement Learning setting.

7. Combine all hyperparameters.

```
hyperparameters = training_setup.set_hyperparameters(
     actions_hyperparameters=actions_hyperparameters,
     agent_hyperparameters=agent_hyperparameters,
     reward_hyperparameters=reward_hyperparameters,
     data_pipeline_hyperparameters=data_pipeline_hyperparameters,
     trainer_hyperparameters=trainer_hyperparameters,
     experiment_directory=experiment_directory,
     hyperparameters_file=None,
     hyperparameters_overrides=hyperparameters_overrides,
 )
```

8. Override the selected reward class. Use cases are expected to differ in what is ingested as reactions to Optimus actions and how the final reward is calculated. There are 2 methods that need to be specified.

	The calculate_reward method is used in the online training pipeline and needs to be written in [TensorFlow](https://github.com/tensorflow/tensorflow). It takes 3 inputs: `actions`, `reactions` and `sign_reward`. The first is the action taken by Optimus for a given observation. The second is a Tensor with data on how an end user reacted to that action. The content of that Tensor is use case specific, e.g. it can be a boolean, e.g. indicating a conversion or it can also contain a range of other measurements relevant to the optimization goal. The last parameter controls if the final reward value should be signed meaning contained between -1 and 1.

	Specifying any reward calculation logic in the calculate_pretrain_reward is needed only when the Optimus model is going to be pretrained and needs to be written with [JAX](https://github.com/google/jax). Optimus would typically be pretrained if there is an already existing dataset with contextual data, actions taken (e.g. manually or by a rule-based algorithm) and outcomes. A separate AI/ML model could be trained on that data to predict rewards based on contextual data and actions taken by Optimus. And those predicted rewards could be used in the calculate_pretrain_reward method (see [Pretraining](#pretraining) section).

```
class DemoReward(base_reward.BaseReward):


 def __init__(
     self,
     *,
     hyperparameters: config_dict.ConfigDict,
 ) -> None:
   super().__init__(
       hyperparameters=hyperparameters,
   )


 def calculate_reward(
     self, actions: tf.Tensor, reactions: tf.Tensor, sign_rewards: bool
 ) -> tf.Tensor:
   reward = tf.where(actions == reactions, 1.0, -1.0)
   if sign_rewards:
     reward = tf.math.sign(reward)
   return reward


 def calculate_pretrain_reward(
     self, batch: jnp.ndarray, actions: jnp.ndarray
 ) -> jnp.ndarray:
   raise NotImplementedError
```

9. Initialize the reward class.

```
reward = DemoReward(
   hyperparameters=hyperparameters,
)
```

# Predictions
1. Initialize the selected agent class using the previously assembled hyperparameters.

```
agent = agents.get_agent(agent_name=_AGENT_NAME)(
   hyperparameters=hyperparameters,
)
```

2. Initialize a model state. It will load a model state from the latest checkpoint from the experiment directory if there’s one or initialize a new state.

```
model_state = base_trainer.initialize_model_state_for_prediction(
   agent=agent,
   hyperparameters=hyperparameters,
)
```

3. Make a prediction using the initialized model state, a prediction seed and contextual data. Note that the contextual data needs to be encoded and in the form of a [JAX](https://github.com/google/jax) or [NumPy](https://github.com/numpy/numpy) array. The action from the prediction output will be a numerical value. In the majority of use cases that value will need to be interpreted before passing to any downstream systems. For instance, a use case could be whether to show or not to show a discount to a user at checkout. 0 could be mapped to “Yes” and 1 to “No”.

```
prediction_output = agent.predict(
   agent_state=model_state,
   batch=jnp.asarray([[0, 1, 2]]),
   prediction_seed=0,
)
print(
   "The most optimal action for this context is:"
   f" {prediction_output.action.item()}"
)
```

# Pretraining
In the Reinforcement Learning setup an agent (a model) makes initial decisions at random. It might not be a preferred option in certain use cases. It is possible to pretrain Optimus if there is an already existing dataset with contextual data, actions taken (e.g. manually or by a rule-based algorithm) and outcomes.

1. Initialize a trainer using the previously assembled hyperparameters and the previously initialized agent and reward classes.

```
trainer = trainers.get_trainer(trainer_name=_TRAINER_NAME)(
   agent=agent,
   reward=reward,
   hyperparameters=hyperparameters,
)
```

2. Run pretraining. Note that the train and evaluation data (optional) need to be already encoded and in the form of a [JAX](https://github.com/google/jax) or [NumPy](https://github.com/numpy/numpy) array.

```
model_state = trainer.pretrain(
   train_dataset=train_data,
   evaluation_dataset=evaluation_data,
)
```

# Training
Optimus needs to be periodically updated to improve over time. The recommended approach is to use programmatically triggered Google [Cloud Platform Vertex AI Pipelines] (https://cloud.google.com/vertex-ai/docs/pipelines/introduction) so the entire process does not require human involvement.

1. Collect an arbitrary number of experiences, which will be used as train data and evaluation data (optional) (see [Data](#data) section). The number of training data examples must be higher than the specified batch size.

2. Initialize a data pipeline using the previously assembled hyperparameters.

```
data_pipeline = data_pipelines.get_data_pipeline(
   data_pipeline_name=_DATA_PIPELINE_NAME
)(hyperparameters=hyperparameters)
```

3. Initialize train and evaluation (optional) pipelines. The train data pipeline requires passing the reward calculation callable from the previously initialized reward class. from the Note that the train and evaluation data (optional) need to be already encoded and in the form of a [JAX](https://github.com/google/jax) or [NumPy](https://github.com/numpy/numpy) array.

```
train_data_pipeline = data_pipeline.train_data_pipeline(
   train_data=train_data, reward_calculation_function=reward.calculate_reward
)
evaluation_data_pipeline = data_pipeline.evaluation_data_pipeline(
   evaluation_data=evaluation_data
)
```

4. Run a training loop. The model will update its knowledge on each train batch and the latest checkpoints will be saved in the experiment directory.

```
for train_batch in iter(train_data_pipeline):
 trainer.train(
     train_data=train_batch, evaluation_dataset=evaluation_data_pipeline
 )
```

# Evaluation
The best way to evaluate the model performance is to provide the evaluation dataset into the trainer (see [Data](#data) section). It can provide insights on how the model performs on a dataset with example where the most optimal actions area already known .If the evaluation rewards are trending upward then the agent is improving and vice versa.

If the dataset is not available, then the best way is to track rewards that an agent receives over the course of its deployment. If the prediction rewards are trending upward then the agent is improving and vice versa.

# Google Cloud Platform implementation
*Iconographic 2. Proposed Optimus implementation on Google Cloud Platform for real-time marketing personalization.*
![Alt text](images/proposed_gcp_implementation_architecture.png?raw=true)

# References

1. Arik, Sercan, et al. “TabNet: Attentive Interpretable Tabular Learning.” arXiv.org, 9 Dec. 2020, arxiv.org/abs/1908.07442.

2. Boudet, Julien, et al. “Beyond Belt-Tightening: How Marketing Can Drive Resiliency during Uncertain Times.” McKinsey & Company, McKinsey & Company, 26 June 2023, www.mckinsey.com/capabilities/growth-marketing-and-sales/our-insights/beyond-belt-tightening-how-marketing-can-drive-resiliency-during-uncertain-times.

3. Sutton, Richard S., et al. Reinforcement Learning: An Introduction. MIT Press Ltd, 2018.



from Utils.utils import set_seed  
from dataset_encoder import prepare_scenarios
from args import ArgsGenerator
from Models.model import ModelContainer
args_generator = ArgsGenerator(dataset_name='CIFAR100', dataset_encoder_name='RN50_clip')
args_model = ModelContainer.Options()
  
set_seed(manualSeed=100)

scenario, scenario_test = prepare_scenarios(args_generator,args_model) 
for task_id, (train_taskset, test_taskset) in enumerate(zip(scenario, scenario_test)):
    x,y,t = train_taskset[0]
    print(f"task {task_id}, x in {x.shape}, y {y}")         
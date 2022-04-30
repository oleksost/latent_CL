import os
import numpy as np 
import torchvision 
from functools import partial
from collections import namedtuple
from torchvision import transforms
from torchvision import transforms as t
from dataclasses import dataclass, field
from typing import Callable, List, Tuple
from continuum.tasks.base import TaskType
from torchvision.transforms.transforms import Compose, ToTensor                                                             
from continuum.datasets import CIFAR10, CIFAR100, InMemoryDataset, CUB200, MNIST, OxfordFlower102, OxfordPet, TinyImageNet200, Car196, SUN397, DomainNet, FGVCAircraft, DTD

project_home = f"{os.environ.get('LARGE_CL_HOME')}" if "LARGE_CL_HOME" in os.environ else "."
cars_taxonomy=["Acura RL Sedan 2012",
                "Acura TL Sedan 2012",
                "Acura TL Type-S 2008",
                "Acura TSX Sedan 2012",
                "Acura Integra Type R 2001",
                "Acura ZDX Hatchback 2012",
                "Aston Martin V8 Vantage Convertible 2012",
                "Aston Martin V8 Vantage Coupe 2012",
                "Aston Martin Virage Convertible 2012",
                "Aston Martin Virage Coupe 2012",
                "Audi RS 4 Convertible 2008",
                "Audi A5 Coupe 2012",
                "Audi TTS Coupe 2012",
                "Audi R8 Coupe 2012",
                "Audi V8 Sedan 1994",
                "Audi 100 Sedan 1994",
                "Audi 100 Wagon 1994",
                "Audi TT Hatchback 2011",
                "Audi S6 Sedan 2011",
                "Audi S5 Convertible 2012",
                "Audi S5 Coupe 2012",
                "Audi S4 Sedan 2012",
                "Audi S4 Sedan 2007",
                "Audi TT RS Coupe 2012",
                "BMW ActiveHybrid 5 Sedan 2012",
                "BMW 1 Series Convertible 2012",
                "BMW 1 Series Coupe 2012",
                "BMW 3 Series Sedan 2012",
                "BMW 3 Series Wagon 2012",
                "BMW 6 Series Convertible 2007",
                "BMW X5 SUV 2007",
                "BMW X6 SUV 2012",
                "BMW M3 Coupe 2012",
                "BMW M5 Sedan 2010",
                "BMW M6 Convertible 2010",
                "BMW X3 SUV 2012",
                "BMW Z4 Convertible 2012",
                "Bentley Continental Supersports Conv. Convertible 2012",
                "Bentley Arnage Sedan 2009",
                "Bentley Mulsanne Sedan 2011",
                "Bentley Continental GT Coupe 2012",
                "Bentley Continental GT Coupe 2007",
                "Bentley Continental Flying Spur Sedan 2007",
                "Bugatti Veyron 16.4 Convertible 2009",
                "Bugatti Veyron 16.4 Coupe 2009",
                "Buick Regal GS 2012",
                "Buick Rainier SUV 2007",
                "Buick Verano Sedan 2012",
                "Buick Enclave SUV 2012",
                "Cadillac CTS-V Sedan 2012",
                "Cadillac SRX SUV 2012",
                "Cadillac Escalade EXT Crew Cab 2007",
                "Chevrolet Silverado 1500 Hybrid Crew Cab 2012",
                "Chevrolet Corvette Convertible 2012",
                "Chevrolet Corvette ZR1 2012",
                "Chevrolet Corvette Ron Fellows Edition Z06 2007",
                "Chevrolet Traverse SUV 2012",
                "Chevrolet Camaro Convertible 2012",
                "Chevrolet HHR SS 2010",
                "Chevrolet Impala Sedan 2007",
                "Chevrolet Tahoe Hybrid SUV 2012",
                "Chevrolet Sonic Sedan 2012",
                "Chevrolet Express Cargo Van 2007",
                "Chevrolet Avalanche Crew Cab 2012",
                "Chevrolet Cobalt SS 2010",
                "Chevrolet Malibu Hybrid Sedan 2010",
                "Chevrolet TrailBlazer SS 2009",
                "Chevrolet Silverado 2500HD Regular Cab 2012",
                "Chevrolet Silverado 1500 Classic Extended Cab 2007",
                "Chevrolet Express Van 2007",
                "Chevrolet Monte Carlo Coupe 2007",
                "Chevrolet Malibu Sedan 2007",
                "Chevrolet Silverado 1500 Extended Cab 2012",
                "Chevrolet Silverado 1500 Regular Cab 2012",
                "Chrysler Aspen SUV 2009",
                "Chrysler Sebring Convertible 2010",
                "Chrysler Town and Country Minivan 2012",
                "Chrysler 300 SRT-8 2010",
                "Chrysler Crossfire Convertible 2008",
                "Chrysler PT Cruiser Convertible 2008",
                "Daewoo Nubira Wagon 2002",
                "Dodge Caliber Wagon 2012",
                "Dodge Caliber Wagon 2007",
                "Dodge Caravan Minivan 1997",
                "Dodge Ram Pickup 3500 Crew Cab 2010",
                "Dodge Ram Pickup 3500 Quad Cab 2009",
                "Dodge Sprinter Cargo Van 2009",
                "Dodge Journey SUV 2012",
                "Dodge Dakota Crew Cab 2010",
                "Dodge Dakota Club Cab 2007",
                "Dodge Magnum Wagon 2008",
                "Dodge Challenger SRT8 2011",
                "Dodge Durango SUV 2012",
                "Dodge Durango SUV 2007",
                "Dodge Charger Sedan 2012",
                "Dodge Charger SRT-8 2009",
                "Eagle Talon Hatchback 1998",
                "FIAT 500 Abarth 2012",
                "FIAT 500 Convertible 2012",
                "Ferrari FF Coupe 2012",
                "Ferrari California Convertible 2012",
                "Ferrari 458 Italia Convertible 2012",
                "Ferrari 458 Italia Coupe 2012",
                "Fisker Karma Sedan 2012",
                "Ford F-450 Super Duty Crew Cab 2012",
                "Ford Mustang Convertible 2007",
                "Ford Freestar Minivan 2007",
                "Ford Expedition EL SUV 2009",
                "Ford Edge SUV 2012",
                "Ford Ranger SuperCab 2011",
                "Ford GT Coupe 2006",
                "Ford F-150 Regular Cab 2012",
                "Ford F-150 Regular Cab 2007",
                "Ford Focus Sedan 2007",
                "Ford E-Series Wagon Van 2012",
                "Ford Fiesta Sedan 2012",
                "GMC Terrain SUV 2012",
                "GMC Savana Van 2012",
                "GMC Yukon Hybrid SUV 2012",
                "GMC Acadia SUV 2012",
                "GMC Canyon Extended Cab 2012",
                "Geo Metro Convertible 1993",
                "HUMMER H3T Crew Cab 2010",
                "HUMMER H2 SUT Crew Cab 2009",
                "Honda Odyssey Minivan 2012",
                "Honda Odyssey Minivan 2007",
                "Honda Accord Coupe 2012",
                "Honda Accord Sedan 2012",
                "Hyundai Veloster Hatchback 2012",
                "Hyundai Santa Fe SUV 2012",
                "Hyundai Tucson SUV 2012",
                "Hyundai Veracruz SUV 2012",
                "Hyundai Sonata Hybrid Sedan 2012",
                "Hyundai Elantra Sedan 2007",
                "Hyundai Accent Sedan 2012",
                "Hyundai Genesis Sedan 2012",
                "Hyundai Sonata Sedan 2012",
                "Hyundai Elantra Touring Hatchback 2012",
                "Hyundai Azera Sedan 2012",
                "Infiniti G Coupe IPL 2012",
                "Infiniti QX56 SUV 2011",
                "Isuzu Ascender SUV 2008",
                "Jaguar XK XKR 2012",
                "Jeep Patriot SUV 2012",
                "Jeep Wrangler SUV 2012",
                "Jeep Liberty SUV 2012",
                "Jeep Grand Cherokee SUV 2012",
                "Jeep Compass SUV 2012",
                "Lamborghini Reventon Coupe 2008",
                "Lamborghini Aventador Coupe 2012",
                "Lamborghini Gallardo LP 570-4 Superleggera 2012",
                "Lamborghini Diablo Coupe 2001",
                "Land Rover Range Rover SUV 2012",
                "Land Rover LR2 SUV 2012",
                "Lincoln Town Car Sedan 2011",
                "MINI Cooper Roadster Convertible 2012",
                "Maybach Landaulet Convertible 2012",
                "Mazda Tribute SUV 2011",
                "McLaren MP4-12C Coupe 2012",
                "Mercedes-Benz 300-Class Convertible 1993",
                "Mercedes-Benz C-Class Sedan 2012",
                "Mercedes-Benz SL-Class Coupe 2009",
                "Mercedes-Benz E-Class Sedan 2012",
                "Mercedes-Benz S-Class Sedan 2012",
                "Mercedes-Benz Sprinter Van 2012",
                "Mitsubishi Lancer Sedan 2012",
                "Nissan Leaf Hatchback 2012",
                "Nissan NV Passenger Van 2012",
                "Nissan Juke Hatchback 2012",
                "Nissan 240SX Coupe 1998",
                "Plymouth Neon Coupe 1999",
                "Porsche Panamera Sedan 2012",
                "Ram C/V Cargo Van Minivan 2012",
                "Rolls-Royce Phantom Drophead Coupe Convertible 2012",
                "Rolls-Royce Ghost Sedan 2012",
                "Rolls-Royce Phantom Sedan 2012",
                "Scion xD Hatchback 2012",
                "Spyker C8 Convertible 2009",
                "Spyker C8 Coupe 2009",
                "Suzuki Aerio Sedan 2007",
                "Suzuki Kizashi Sedan 2012",
                "Suzuki SX4 Hatchback 2012",
                "Suzuki SX4 Sedan 2012",
                "Tesla Model S Sedan 2012",
                "Toyota Sequoia SUV 2012",
                "Toyota Camry Sedan 2012",
                "Toyota Corolla Sedan 2012",
                "Toyota 4Runner SUV 2012",
                "Volkswagen Golf Hatchback 2012",
                "Volkswagen Golf Hatchback 1991",
                "Volkswagen Beetle Hatchback 2012",
                "Volvo C30 Hatchback 2012",
                "Volvo 240 Sedan 1993",
                "Volvo XC90 SUV 2007",
                "smart fortwo Convertible 2012"]

flowers_taxonomy = ['pink primrose', 'hard-leaved pocket orchid', 'canterbury bells', 'sweet pea', 'english marigold', 'tiger lily', 'moon orchid', 'bird of paradise', 'monkshood', 'globe thistle', 'snapdragon', "colt's foot", 'king protea', 'spear thistle', 'yellow iris', 'globe-flower', 'purple coneflower', 'peruvian lily', 'balloon flower', 'giant white arum lily', 'fire lily', 'pincushion flower', 'fritillary', 'red ginger', 'grape hyacinth', 'corn poppy', 'prince of wales feathers', 'stemless gentian', 'artichoke', 'sweet william', 'carnation', 'garden phlox', 'love in the mist', 'mexican aster', 'alpine sea holly', 'ruby-lipped cattleya', 'cape flower', 'great masterwort', 'siam tulip', 'lenten rose', 'barbeton daisy', 'daffodil', 'sword lily', 'poinsettia', 'bolero deep blue', 'wallflower', 'marigold', 'buttercup', 'oxeye daisy', 'common dandelion', 'petunia', 'wild pansy', 'primula', 'sunflower', 'pelargonium', 'bishop of llandaff', 'gaura', 'geranium', 'orange dahlia', 'pink-yellow dahlia?', 'cautleya spicata', 'japanese anemone', 'black-eyed susan', 'silverbush', 'californian poppy', 'osteospermum', 'spring crocus', 'bearded iris', 'windflower', 'tree poppy', 'gazania', 'azalea', 'water lily', 'rose', 'thorn apple', 'morning glory', 'passion flower', 'lotus', 'toad lily', 'anthurium', 'frangipani', 'clematis', 'hibiscus', 'columbine', 'desert-rose', 'tree mallow', 'magnolia', 'cyclamen ', 'watercress', 'canna lily', 'hippeastrum ', 'bee balm', 'ball moss', 'foxglove', 'bougainvillea', 'camellia', 'mallow', 'mexican petunia', 'bromelia', 'blanket flower', 'trumpet creeper', 'blackberry lily']

CUB_taxonomy=["Black_footed_Albatross",
                    "Laysan_Albatross",
                    "Sooty_Albatross",
                    "Groove_billed_Ani",
                    "Crested_Auklet",
                    "Least_Auklet",
                    "Parakeet_Auklet",
                    "Rhinoceros_Auklet",
                    "Brewer_Blackbird",
                    "Red_winged_Blackbird",
                    "Rusty_Blackbird",
                    "Yellow_headed_Blackbird",
                    "Bobolink",
                    "Indigo_Bunting",
                    "Lazuli_Bunting",
                    "Painted_Bunting",
                    "Cardinal",
                    "Spotted_Catbird",
                    "Gray_Catbird",
                    "Yellow_breasted_Chat",
                    "Eastern_Towhee",
                    "Chuck_will_Widow",
                    "Brandt_Cormorant",
                    "Red_faced_Cormorant",
                    "Pelagic_Cormorant",
                    "Bronzed_Cowbird",
                    "Shiny_Cowbird",
                    "Brown_Creeper",
                    "American_Crow",
                    "Fish_Crow",
                    "Black_billed_Cuckoo",
                    "Mangrove_Cuckoo",
                    "Yellow_billed_Cuckoo",
                    "Gray_crowned_Rosy_Finch",
                    "Purple_Finch",
                    "Northern_Flicker",
                    "Acadian_Flycatcher",
                    "Great_Crested_Flycatcher",
                    "Least_Flycatcher",
                    "Olive_sided_Flycatcher",
                    "Scissor_tailed_Flycatcher",
                    "Vermilion_Flycatcher",
                    "Yellow_bellied_Flycatcher",
                    "Frigatebird",
                    "Northern_Fulmar",
                    "Gadwall",
                    "American_Goldfinch",
                    "European_Goldfinch",
                    "Boat_tailed_Grackle",
                    "Eared_Grebe",
                    "Horned_Grebe",
                    "Pied_billed_Grebe",
                    "Western_Grebe",
                    "Blue_Grosbeak",
                    "Evening_Grosbeak",
                    "Pine_Grosbeak",
                    "Rose_breasted_Grosbeak",
                    "Pigeon_Guillemot",
                    "California_Gull",
                    "Glaucous_winged_Gull",
                    "Heermann_Gull",
                    "Herring_Gull",
                    "Ivory_Gull",
                    "Ring_billed_Gull",
                    "Slaty_backed_Gull",
                    "Western_Gull",
                    "Anna_Hummingbird",
                    "Ruby_throated_Hummingbird",
                    "Rufous_Hummingbird",
                    "Green_Violetear",
                    "Long_tailed_Jaeger",
                    "Pomarine_Jaeger",
                    "Blue_Jay",
                    "Florida_Jay",
                    "Green_Jay",
                    "Dark_eyed_Junco",
                    "Tropical_Kingbird",
                    "Gray_Kingbird",
                    "Belted_Kingfisher",
                    "Green_Kingfisher",
                    "Pied_Kingfisher",
                    "Ringed_Kingfisher",
                    "White_breasted_Kingfisher",
                    "Red_legged_Kittiwake",
                    "Horned_Lark",
                    "Pacific_Loon",
                    "Mallard",
                    "Western_Meadowlark",
                    "Hooded_Merganser",
                    "Red_breasted_Merganser",
                    "Mockingbird",
                    "Nighthawk",
                    "Clark_Nutcracker",
                    "White_breasted_Nuthatch",
                    "Baltimore_Oriole",
                    "Hooded_Oriole",
                    "Orchard_Oriole",
                    "Scott_Oriole",
                    "Ovenbird",
                    "Brown_Pelican",
                    "White_Pelican",
                    "Western_Wood_Pewee",
                    "Sayornis",
                    "American_Pipit",
                    "Whip_poor_Will",
                    "Horned_Puffin",
                    "Common_Raven",
                    "White_necked_Raven",
                    "American_Redstart",
                    "Geococcyx",
                    "Loggerhead_Shrike",
                    "Great_Grey_Shrike",
                    "Baird_Sparrow",
                    "Black_throated_Sparrow",
                    "Brewer_Sparrow",
                    "Chipping_Sparrow",
                    "Clay_colored_Sparrow",
                    "House_Sparrow",
                    "Field_Sparrow",
                    "Fox_Sparrow",
                    "Grasshopper_Sparrow",
                    "Harris_Sparrow",
                    "Henslow_Sparrow",
                    "Le_Conte_Sparrow",
                    "Lincoln_Sparrow",
                    "Nelson_Sharp_tailed_Sparrow",
                    "Savannah_Sparrow",
                    "Seaside_Sparrow",
                    "Song_Sparrow",
                    "Tree_Sparrow",
                    "Vesper_Sparrow",
                    "White_crowned_Sparrow",
                    "White_throated_Sparrow",
                    "Cape_Glossy_Starling",
                    "Bank_Swallow",
                    "Barn_Swallow",
                    "Cliff_Swallow",
                    "Tree_Swallow",
                    "Scarlet_Tanager",
                    "Summer_Tanager",
                    "Artic_Tern",
                    "Black_Tern",
                    "Caspian_Tern",
                    "Common_Tern",
                    "Elegant_Tern",
                    "Forsters_Tern",
                    "Least_Tern",
                    "Green_tailed_Towhee",
                    "Brown_Thrasher",
                    "Sage_Thrasher",
                    "Black_capped_Vireo",
                    "Blue_headed_Vireo",
                    "Philadelphia_Vireo",
                    "Red_eyed_Vireo",
                    "Warbling_Vireo",
                    "White_eyed_Vireo",
                    "Yellow_throated_Vireo",
                    "Bay_breasted_Warbler",
                    "Black_and_white_Warbler",
                    "Black_throated_Blue_Warbler",
                    "Blue_winged_Warbler",
                    "Canada_Warbler",
                    "Cape_May_Warbler",
                    "Cerulean_Warbler",
                    "Chestnut_sided_Warbler",
                    "Golden_winged_Warbler",
                    "Hooded_Warbler",
                    "Kentucky_Warbler",
                    "Magnolia_Warbler",
                    "Mourning_Warbler",
                    "Myrtle_Warbler",
                    "Nashville_Warbler",
                    "Orange_crowned_Warbler",
                    "Palm_Warbler",
                    "Pine_Warbler",
                    "Prairie_Warbler",
                    "Prothonotary_Warbler",
                    "Swainson_Warbler",
                    "Tennessee_Warbler",
                    "Wilson_Warbler",
                    "Worm_eating_Warbler",
                    "Yellow_Warbler",
                    "Northern_Waterthrush",
                    "Louisiana_Waterthrush",
                    "Bohemian_Waxwing",
                    "Cedar_Waxwing",
                    "American_Three_toed_Woodpecker",
                    "Pileated_Woodpecker",
                    "Red_bellied_Woodpecker",
                    "Red_cockaded_Woodpecker",
                    "Red_headed_Woodpecker",
                    "Downy_Woodpecker",
                    "Bewick_Wren",
                    "Cactus_Wren",
                    "Carolina_Wren",
                    "House_Wren",
                    "Marsh_Wren",
                    "Rock_Wren",
                    "Winter_Wren",
                    "Common_Yellowthroat"]
CIFAR10_taxonomy = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog',
                        'frog', 'horse', 'ship', 'truck']
CIFAR100_taxonomy = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']

pets_taxonomy = [
                "Abyssinian"
                "Bengal",                   
                "Birman",                       
                "Bombay",    
                "British Shorthair",    
                "Egyptian Mau",    
                "Maine Coon",    
                "Persian",    
                "Ragdoll",    
                "Russian Blue",    
                "Siamese",    
                "Sphynx",    
                "american bulldog",    
                "american pit bull terrier",    
                "basset hound",    
                "beagle",    
                "boxer",    
                "chihuahua",    
                "english cocker spaniel",    
                "english setter",    
                "german shorthaired",    
                "great pyrenees",    
                "havanese",    
                "japanese chin",    
                "keeshond",
                "leonberger",    
                "miniature pinscher",    
                "newfoundland",    
                "pomeranian",    
                "pug",    
                "saint bernard",
                "samoyed",
                "scottish terrier",
                "shiba inu",
                "staffordshire bull terrier",
                "wheaten terrier",
                "yorkshire terrier"
                ]

def create_increments(n_classes, n_tasks):
    #from https://stackoverflow.com/questions/20348717/algo-for-dividing-a-number-into-almost-equal-whole-numbers
    return [n_classes // n_tasks + (1 if x < n_classes % n_tasks else 0)  for x in range (n_tasks)]

def prepare_dataset(dataset, path):     
    dataset_train = dataset(data_path=path, train=True, download=True)
    dataset_test = dataset(data_path=path, train=False, download=True)
    return dataset_train, dataset_test

def prepare_dataset_big(list_datasets, path): 
    list_instanciate_datasets_train = []
    list_instanciate_datasets_test = []
    for i, dataset_class in enumerate(list_datasets):
        ds_train, ds_test = dataset_class.get_datasets(path)
        list_instanciate_datasets_train.append(ds_train)
        list_instanciate_datasets_test.append(ds_test)
        # list_instanciate_datasets_train.append(dataset_class(path, train=True, download=True))
        # list_instanciate_datasets_test.append(dataset_class(path, train=False, download=True))
    return list_instanciate_datasets_train, list_instanciate_datasets_test

   
dataset_tuple = namedtuple('DatasetTuple',['get_datasets', 'dataset_info'])



@dataclass
class Datasets_Info:
    # name:str
    n_tasks:int
    resolution: int
    n_classes: int = 0
    increment: int = 0  
    tasktype: str = None        
    append_transfrom: Callable = None
    data_path: str =f'{project_home}/Data/'
    taxonomy: List = field(default_factory=list)    
    list_datasets: List = field(default_factory=list)
    transformations: List = field(default_factory=list)
    transformations_val: List = field(default_factory=list)
    size: List = field(default_factory= list)
    def __post_init__(self):
        self.name=self.__class__.__name__.split('_')[0]
        if self.n_classes%self.n_tasks!=0:
            self.increment=create_increments(self.n_classes, self.n_tasks) 
            if self.increment!=0:
                assert sum(self.increment)==self.n_classes
        self.n_classes_per_task=self.increment #[int(self.n_classes/self.n_tasks)]*self.n_tasks

@dataclass
class CIFAR100_info(Datasets_Info):
    n_classes: int = 100
    def __post_init__(self):
        super().__post_init__()
        self.taxonomy = CIFAR100_taxonomy #get_cifar100_classes(f'{project_home}/Data/') #CIFAR100_taxonomy/
        self.size = [3,100,100]
        if self.resolution is not None:
            self.size = [3,self.resolution,self.resolution]
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        ##################
        #use these transfroms if encoder does not provide custom transfroms
        # self.transformations = [#  transforms.RandomHorizontalFlip(),
        #                 #  transforms.RandomCrop(32, padding=4),
        #                 #  transforms.ToTensor(),
        #                 #  transforms.Normalize(mean=[n/255.
        #                 #     for n in [129.3, 124.1, 112.4]], std=[n/255. for n in [68.2,  65.4,  
        #         # transforms.RandomHorizontalFlip(),
        #         transforms.RandomCrop((self.size[-1], self.size[-1])),
        #         transforms.ToTensor(),
        #         transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))]
        # self.transformations_val = [transforms.Resize((self.size[-1], self.size[-1])), transforms.ToTensor(),transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))]  
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.size = [3,100,100]
        if self.resolution is not None:
            self.size = [3,self.resolution,self.resolution]     
        #use continuum transfroms
        self.transformations = [
            transforms.Resize([self.size[-1], self.size[-1]]),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)]
            # transforms.Normalize(mean, std)]
        
        self.transformations_val = [   
            transforms.Resize([self.size[-1], self.size[-1]]),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)]
            # transforms.Normalize(mean, std)]


@dataclass
class CIFAR10_info(Datasets_Info):
    n_classes: int = 10
    def __post_init__(self):
        super().__post_init__()               
        self.taxonomy = CIFAR10_taxonomy
        # self.size = [3,32,32]
        # self.transformations = [#  transforms.RandomHorizontalFlip(),
        #                 #  transforms.RandomCrop(32, padding=4),
        #                 #  transforms.ToTensor(),
        #                 #  transforms.Normalize(mean=[n/255.
        #                 #     for n in [129.3, 124.1, 112.4]], std=[n/255. for n in [68.2,  65.4,  
        #         transforms.RandomHorizontalFlip(),
        #         transforms.RandomCrop(32, 4),
        #         transforms.ToTensor(),
        #         # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        #         transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                 std=[0.229, 0.224, 0.225])]

        # self.transformations_val = [transforms.ToTensor(),
        #         # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        #         transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                 std=[0.229, 0.224, 0.225])]
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.size = [3,100,100]
        if self.resolution is not None:
            self.size = [3,self.resolution,self.resolution]        
        #use continuum transfroms
        self.transformations = [
            transforms.Resize([self.size[-1], self.size[-1]]),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)]
            # transforms.Normalize(mean, std)]
        
        self.transformations_val = [   
            transforms.Resize([self.size[-1], self.size[-1]]),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)]
            # transforms.Normalize(mean, std)]


@dataclass
class CUB200_info(Datasets_Info):
    n_classes: int = 200
    def __post_init__(self):
        super().__post_init__()
        self.taxonomy = CUB_taxonomy
        self.tasktype = TaskType.IMAGE_PATH
        self.size = [3,224,224]
        if self.resolution is not None:
            self.size = [3,self.resolution,self.resolution]
        # self.transformations = [
        #     transforms.Resize((self.size, self.size), Image.BILINEAR),
        #     # transforms.RandomHorizontalFlip(),  # only if train
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]

        # self.transformations_val = [
        #             transforms.Resize((224, 224), Image.BILINEAR),
        #             # transforms.RandomHorizontalFlip(),  # only if train
        #             transforms.ToTensor(),
        #             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        horizontal_flip = 0.5
        
        self.transformations = [
            transforms.Resize([self.size[-1], self.size[-1]]),     
            transforms.RandomHorizontalFlip(horizontal_flip) if horizontal_flip is not None else None,
            transforms.ToTensor(),
            transforms.Normalize(mean, std)]
        
        self.transformations_val = [
            transforms.Resize([self.size[-1], self.size[-1]]),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)]

        # if 'SCRATCH' in os.environ.keys():
        #     if os.path.isdir(os.path.join(os.environ['HOME'],"scratch/Data/")):
        #         self.data_path=f"{os.environ['SCRATCH']}/Data/CUB_200_2011/"
        #     print(self.data_path)

@dataclass
class MNIST_info(Datasets_Info):
    n_classes: int = 10
    def __post_init__(self):
        super().__post_init__()
        self.taxonomy = torchvision.datasets.MNIST.classes
        self.size = [1,28,28]
        mean = [0.1307,0.1307,0.1307]
        std = [0.3081,0.3081,0.3081] 
        self.append_transfrom = transforms.Lambda(lambda x: x.repeat(3, 1, 1)) #transfrom to RGBm this transfrom is papended to any transfromations used in the TasSet
        #use continuum transfroms
        self.transformations = None #[
            # transforms.ToTensor(),
            # transforms.Lambda(lambda x: x.repeat(3, 1, 1)), #transfrom to RGB
            # transforms.Normalize(mean, std)]
        
        self.transformations_val = None #[
            # transforms.ToTensor(),
            # transforms.Lambda(lambda x: x.repeat(3, 1, 1)), #transfrom to RGB
            # transforms.Normalize(mean, std)]



@dataclass   
class OxfordFlower102_info(Datasets_Info):
    n_classes: int = 102
    def __post_init__(self):
        super().__post_init__()
        mean = [0.5,0.5,0.5]
        std = [0.5,0.5,0.5]   
        self.taxonomy = flowers_taxonomy
        self.size = [3,100,100]
        if self.resolution is not None:
            self.size = [3,self.resolution,self.resolution]
        #use continuum transfroms
        self.transformations = [
            transforms.Resize([self.size[-1], self.size[-1]]),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)]
            # transforms.Normalize(mean, std)]
        
        self.transformations_val = [   
            transforms.Resize([self.size[-1], self.size[-1]]),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)]
            # transforms.Normalize(mean, std)]
@dataclass
class OxfordPet_info(Datasets_Info):
    n_classes: int = 37
    def __post_init__(self):
        super().__post_init__()
        mean = [0.5,0.5,0.5]
        std = [0.5,0.5,0.5]        
        self.taxonomy = pets_taxonomy
        self.size = [3,100,100]
        if self.resolution is not None:
            self.size = [3,self.resolution,self.resolution]
        #use continuum transfroms
        self.transformations = [
            transforms.Resize([self.size[-1], self.size[-1]]),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)]
            # transforms.Normalize(mean, std)]
        
        self.transformations_val = [   
            transforms.Resize([self.size[-1], self.size[-1]]),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)]
            # transforms.Normalize(mean, std)]

@dataclass
class TinyImageNet200_info(Datasets_Info):
    n_classes: int = 200
    def __post_init__(self):
        super().__post_init__()
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.size = [3,224,224]
        if self.resolution is not None:
            self.size = [3,self.resolution,self.resolution]
        self.taxonomy=None
        #use continuum transfroms
        self.transformations = [       
            transforms.Resize([self.size[-1], self.size[-1]]),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)]
            # transforms.Normalize(mean, std)]        
        self.transformations_val = [   
            transforms.Resize([self.size[-1], self.size[-1]]),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)]
            # transforms.Normalize(mean, std)]

class Car196_info(Datasets_Info):
    n_classes: int = 196
    def __post_init__(self):
        self.n_classes=196
        super().__post_init__()
        mean = [0.5,0.5,0.5]
        std = [0.5,0.5,0.5]         
        self.taxonomy = cars_taxonomy 
        self.size = [3,100,100]
        if self.resolution is not None:
            self.size = [3,self.resolution,self.resolution]
        #use continuum transfroms
        self.transformations = [
            transforms.Resize([self.size[-1], self.size[-1]]),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)]
            # transforms.Normalize(mean, std)]
        
        self.transformations_val = [   
            transforms.Resize([self.size[-1], self.size[-1]]),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)]
            # transforms.Normalize(mean, std)]


class SUN397_info(Datasets_Info):
    n_classes: int = 397
    def __post_init__(self):
        self.n_classes=397
        super().__post_init__()
        mean = [0.5,0.5,0.5]
        std = [0.5,0.5,0.5]         
        self.taxonomy = None 
        self.size = [3,400,400]
        if self.resolution is not None:
            self.size = [3,self.resolution,self.resolution]
        #use continuum transfroms
        self.transformations = [
            transforms.Resize([self.size[-1], self.size[-1]]),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)]
            # transforms.Normalize(mean, std)]
        
        self.transformations_val = [   
            transforms.Resize([self.size[-1], self.size[-1]]),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)]
            # transforms.Normalize(mean, std)]


class DomainNet_info(Datasets_Info):
    n_classes: int = 345
    def __post_init__(self):
        self.n_classes=345
        super().__post_init__()
        mean = [0.5,0.5,0.5]
        std = [0.5,0.5,0.5]       
        self.increment=0  
        self.taxonomy = None 
        self.size = [3,100,100]
        #use continuum transfroms
        self.n_tasks=6
        self.transformations = [
            transforms.Resize([self.size[-1], self.size[-1]]),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)]
            # transforms.Normalize(mean, std)]
        
        self.transformations_val = [   
            transforms.Resize([self.size[-1], self.size[-1]]),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)]
            # transforms.Normalize(mean, std)]
    

class DTD_info(Datasets_Info):
    n_classes: int = 47
    def __post_init__(self):
        self.n_classes=47
        super().__post_init__()
        mean = [0.5,0.5,0.5]
        std = [0.5,0.5,0.5]
        self.taxonomy = None 
        self.size = [3,100,100]
        if self.resolution is not None:
            self.size = [3,self.resolution,self.resolution]
        #use continuum transfroms
        self.transformations = [
            transforms.Resize([self.size[-1], self.size[-1]]),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)]
            # transforms.Normalize(mean, std)]
        
        self.transformations_val = [   
            transforms.Resize([self.size[-1], self.size[-1]]),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)]
   

class FGVCAircraft_info(Datasets_Info):
    n_classes: int = 100
    def __post_init__(self):
        self.n_classes=100
        super().__post_init__()
        mean = [0.5,0.5,0.5]
        std = [0.5,0.5,0.5]   
        self.taxonomy = None 
        self.size = [3,100,100]
        if self.resolution is not None:
            self.size = [3,self.resolution,self.resolution]
        self.transformations = [
            transforms.Resize([self.size[-1], self.size[-1]]),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)]
            # transforms.Normalize(mean, std)]
        
        self.transformations_val = [   
            transforms.Resize([self.size[-1], self.size[-1]]),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)]

@dataclass   
class Big_info(Datasets_Info):
    def __post_init__(self):
        super().__post_init__()  
        # self.list_datasets = [datasets['CUB200'],datasets['Car196'], datasets['CIFAR100'], datasets['DTD'], datasets['OxfordPet'], datasets['FGVCAircraft']]
        self.list_datasets = [datasets['Car196'], datasets['CIFAR100'], datasets['DTD'], datasets['OxfordPet'], datasets['FGVCAircraft']]
        
        self.n_tasks=len(self.list_datasets)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.size = [3,224,224]
        if self.resolution is not None:
            self.size = [3,self.resolution,self.resolution]
        #use continuum transfroms
        self.transformations = [
            transforms.Resize([self.size[-1], self.size[-1]]),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)]
            # transforms.Normalize(mean, std)]
        
        self.transformations_val = [   
            transforms.Resize([self.size[-1], self.size[-1]]),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)]
            # transforms.Normalize(mean, std)]
        self.taxonomy=[]
        self.increment=[]     
        dataset_list = []
        self.n_classes = 0    
        for _, dataset in enumerate(self.list_datasets): 
            dataset = dataset._replace(dataset_info=dataset.dataset_info(n_tasks=1, resolution=None))
            self.increment.append(dataset.dataset_info.n_classes)
            self.n_classes+=dataset.dataset_info.n_classes
            if self.resolution is not None:
                dataset.dataset_info.transformations=self.transformations
                dataset.dataset_info.transformations_val=self.transformations_val
            if dataset.dataset_info.taxonomy is None:
                self.taxonomy=None
            else:
                if self.taxonomy is not None:
                    self.taxonomy+=dataset.dataset_info.taxonomy
            dataset_list.append(dataset)
        self.list_datasets=dataset_list
        self.n_tasks=len(self.increment)
        self.n_classes=self.increment

@dataclass   
class BigAugm_info(Datasets_Info):
    def __post_init__(self):
        super().__post_init__()   
        # self.list_datasets = [datasets['CIFAR10'],datasets['CIFAR10'], datasets['CIFAR10'], datasets['CIFAR10']]#, datasets['OxfordPet'], datasets['FGVCAircraft']]

        self.list_datasets = [datasets['CUB200'],datasets['Car196'], datasets['CIFAR100'], datasets['DTD'], datasets['OxfordPet'], datasets['FGVCAircraft']]
        
        self.n_tasks=len(self.list_datasets)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.size = [3,224,224]
        #use continuum transfroms
        transformations = [
            transforms.Resize([self.size[-1], self.size[-1]]),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)]
            # transforms.Normalize(mean, std)]
        
        transformations_val = [   
            transforms.Resize([self.size[-1], self.size[-1]]),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)]
            # transforms.Normalize(mean, std)]

        
        transform_blur = [Compose(transformations), t.GaussianBlur((1,11),3)]         
        transform_night = [Compose(transformations), lambda x: t.functional.adjust_gamma(x, gamma=3) , t.ColorJitter(brightness=[0.1,0.1])]
        transform_pencil = [Compose(transformations), t.ColorJitter(saturation=[0,0], contrast=[1.5,1.5], brightness=[2,2])] # PencilSketch(args.input_size[1],args.input_size[2]) #pencil #t.Compose([t.ColorJitter(saturation=[0,0], contrast=[5,5], brightness=[3,5])])
        
        self.transformations = [transformations,transform_blur,transform_night,transform_pencil]
        self.transformations_val = self.transformations

        self.taxonomy=[]
        self.increment=[]     
        dataset_list = []
        self.n_classes = 0    
        for _, dataset in enumerate(self.list_datasets):
            dataset = dataset._replace(dataset_info=dataset.dataset_info(n_tasks=1))
            self.increment.append(dataset.dataset_info.n_classes)
            self.n_classes+=dataset.dataset_info.n_classes
            dataset.dataset_info.transformations=self.transformations
            dataset.dataset_info.transformations_val=self.transformations_val
            if dataset.dataset_info.taxonomy is None:
                self.taxonomy=None
            else:
                if self.taxonomy is not None:
                    self.taxonomy+=dataset.dataset_info.taxonomy
            dataset_list.append(dataset)
        self.list_datasets=dataset_list
   
datasets={    
    'Big': dataset_tuple(partial(prepare_dataset_big ), Big_info),
    'BigAugm': dataset_tuple(partial(prepare_dataset_big ), BigAugm_info), 
    'CIFAR10': dataset_tuple(partial(prepare_dataset, CIFAR10), CIFAR10_info), #name='CIFAR10')),
    'CIFAR100': dataset_tuple(partial(prepare_dataset, CIFAR100), CIFAR100_info), #name='CIFAR100')),
    'CUB200': dataset_tuple(partial(prepare_dataset, CUB200), CUB200_info), #name='CUB200')),
    'TinyImageNet200':dataset_tuple(partial(prepare_dataset, TinyImageNet200), TinyImageNet200_info), #name='TinyImageNet200')),
    'Car196': dataset_tuple(partial(prepare_dataset, Car196), Car196_info),
    'FGVCAircraft': dataset_tuple(partial(prepare_dataset, FGVCAircraft), FGVCAircraft_info),
    'DTD': dataset_tuple(partial(prepare_dataset, DTD), DTD_info),
    'DomainNet': dataset_tuple(partial(prepare_dataset, DomainNet), DomainNet_info), #name='OxfordPet')),
    'OxfordFlower102': dataset_tuple(partial(prepare_dataset, OxfordFlower102), OxfordFlower102_info), #name='OxfordFlower102')),
    'OxfordPet': dataset_tuple(partial(prepare_dataset, OxfordPet), OxfordPet_info), #name='OxfordPet')),
    'SUN397': dataset_tuple(partial(prepare_dataset, SUN397), SUN397_info), #name='OxfordPet')),
}
import pandas as pd
!pip install -U pyabsa
from google.colab import drive
drive.mount('/content/drive/')
from pyabsa.functional import ATEPCModelList
from pyabsa.functional import Trainer, ATEPCTrainer
from pyabsa.functional import ABSADatasetList
from pyabsa.functional import ATEPCConfigManager

atepc_config = ATEPCConfigManager.get_atepc_config_english()

atepc_config.pretrained_bert = 'microsoft/deberta-v3-base'
atepc_config.model = ATEPCModelList.FAST_LCF_ATEPC
dataset_path = ABSADatasetList.Restaurant14
# or your local dataset: dataset_path = 'your local dataset path'

aspect_extractor = ATEPCTrainer(config=atepc_config,
                                dataset=dataset_path,
                                from_checkpoint='',  # set checkpoint to train on the checkpoint.
                                checkpoint_save_mode=1,
                                auto_device=True
                                ).load_trained_model()
df = pd.read_csv('/content/drive/MyDrive/reviews.csv',sep='\t')
df.head()
from pyabsa.functional import ABSADatasetList
from pyabsa.functional import ATEPCCheckpointManager

examples = df['Review']

inference_source = ABSADatasetList.Restaurant14
aspect_extractor = ATEPCCheckpointManager.get_aspect_extractor(checkpoint='multilingual')
atepc_result = aspect_extractor.extract_aspect(inference_source=inference_source,
                                               save_result=True,
                                               print_result=True,  # print the result
                                               pred_sentiment=True,  # Predict the sentiment of extracted aspect terms
                                               )
from pyabsa import available_checkpoints

checkpoint_map = available_checkpoints()
result_list = []
for i in range(len(atepc_result)):
  asp = atepc_result[i]['aspect']
  sen = atepc_result[i]['sentiment']
  result_list.append([asp,sen,i,df['Name'][i]])
print(result_list)
result_df = pd.DataFrame(result_list,columns=["Aspect", "Sentiment","review_ID","Name"])
print(result_df)

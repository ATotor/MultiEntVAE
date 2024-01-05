import argparse

def parse_arguments():
   
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size',type=int,default=32)
    parser.add_argument('--learning_rate',type=float,default=0.0001)
    parser.add_argument('--num_epochs', type=int,default=100)
    parser.add_argument('--model_name',type=str,default="conv_auto_enco_full_data")
    
    
    #parser.add_argument('--train_sound_dir',type=str,default="/data/atiam_ml_mvae/nsynth-train/audio")
    parser.add_argument('--train_sound_dir',type=str,default="C:/Users/alexa/OneDrive/Bureau/audio_train")
    #parser.add_argument('--test_sound_dir',type=str,default="/data/atiam_ml_mvae/nsynth_valid/audio")
    parser.add_argument('--test_sound_dir',type=str,default="C:/Users/alexa/OneDrive/Bureau/audio")
    #parser.add_argument('--train_json_file',type=str,default="/data/atiam_ml_mvae/nsynth-train/examples.json")
    parser.add_argument('--train_json_file',type=str,default="C:/Users/alexa/OneDrive/Bureau/examples_train.json")
    #parser.add_argument('--test_json_file',type=str,default="/data/atiam_ml_mvae/nsynth_valid/examples.json")
    parser.add_argument('--test_json_file',type=str,default="C:/Users/alexa/OneDrive/Bureau/examples.json")
    #parser.add_argument('--tensorboard_dir',type=str,default="/net/inavouable/u.salles/philippon/tensorboard_auto_enco")
    parser.add_argument('--tensorboard_dir',type=str,default="C:/Users/alexa/OneDrive/Bureau/devoir projet ATIAM")
    #parser.add_argument('--model_dir',type=str,default="/net/inavouable/u.salles/philippon/model_auto_enco")
    parser.add_argument('--model_dir',type=str,default="C:/Users/alexa/OneDrive/Bureau/devoir projet ATIAM")
    args = parser.parse_args()
    return vars(args)

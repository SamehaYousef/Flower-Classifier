import tensorflow as tf
import tensorflow_hub as hub
import json
import argparse 
import utility_functions as uf

def get_results(image_path, load_model, top_k, all_class_labels):    
    probs, classes = uf.predict(image_path, load_model, int(top_k), all_class_labels)
    print(image_path)
    print(probs)
    
    print("\n\nThe flower most likely belongs to the {:} class with a probability of {:.4f}".format(classes[0], probs[0]))
    return

if __name__ == "__main__" :
    parser = argparse.ArgumentParser(description = "Predicts the flower's class")
    
    parser.add_argument("--top_k", help= "Return the top K most likely classes of the flower", required = False, default = 5)
    parser.add_argument("--category_names", help= "Path to a JSON file mapping labels to flower names", required = False, default = "label_map.json")
    parser.add_argument("--image_path", help= "Image Path", required = False, default = './test_images/cautleya_spicata.jpg')
    parser.add_argument("--model", help= "Model Path", required = False, default = 'best_model.h5')
    args = parser.parse_args()
    get_results(args.image_path, args.model, args.top_k, args.category_names)
    

    
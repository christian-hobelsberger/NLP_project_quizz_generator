# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 23:38:04 2024

@author: Rutger Lemein
"""

from transformers import BartTokenizer, BartForConditionalGeneration

# Define the path where the model is saved
save_path_model = "C:/Users/Rutger Lemein/Downloads/Single_Fine_Tuned_System/Results/results_CL/model_CL_test"

# Load the model from the saved path
model = BartForConditionalGeneration.from_pretrained(save_path_model)
model_name = "facebook/bart-base"  # You can change the model name as needed
tokenizer = BartTokenizer.from_pretrained(model_name)

def generate_QA(input_text):
    # Tokenize the input text
    input_ids = tokenizer(input_text, return_tensors="pt")["input_ids"]

    # Generate outputs
    outputs = model.generate(input_ids=input_ids, max_length=1024)

    # Decode the generated outputs
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Print the generated text
    print(generated_text)
    
    return generated_text
    
def main():
    # Input text
    input_text1 = "Support:  Candy making is a delightful culinary art that involves transforming simple ingredients into delicious treats loved by people of all ages. The process typically begins with the selection and preparation of ingredients such as sugar, corn syrup, flavorings, and colorings. These ingredients are carefully measured and mixed together in precise proportions to create the desired taste and texture. Once the ingredients are mixed, they are heated to a high temperature in a large pot or kettle. This process, known as cooking or boiling, is essential for dissolving the sugar and forming a smooth, uniform mixture. The temperature must be carefully controlled to ensure that the candy reaches the correct consistency without burning. After cooking, the hot candy mixture is poured onto a flat surface, such as a marble slab or a metal table, to cool and solidify. As it cools, the candy is shaped and molded into various forms, such as bars, drops, or sheets. This step requires skill and precision to achieve the desired shape and thickness."
    input_text2 = "Support:  Cadmium is a chemical element; it has symbol Cd and atomic number 48. This soft, silvery-white metal is chemically similar to the two other stable metals in group 12, zinc and mercury. Like zinc, it demonstrates oxidation state +2 in most of its compounds, and like mercury, it has a lower melting point than the transition metals in groups 3 through 11. Cadmium and its congeners in group 12 are often not considered transition metals, in that they do not have partly filled d or f electron shells in the elemental or common oxidation states."
    input_text3 = "Support:  A black hole is a region of spacetime where gravity is so strong that nothing, including light and other electromagnetic waves, is capable of possessing enough energy to escape it. Einstein's theory of general relativity predicts that a sufficiently compact mass can deform spacetime to form a black hole. The boundary of no escape is called the event horizon. A black hole has a great effect on the fate and circumstances of an object crossing it, but it has no locally detectable features according to general relativity.[5] In many ways, a black hole acts like an ideal black body, as it reflects no light."
    input_text4 = "Support:  Cosplay, a portmanteau of \"costume play\", is an activity and performance art in which participants called cosplayers wear costumes and fashion accessories to represent a specific character. Cosplayers often interact to create a subculture, and a broader use of the term \"cosplay\" applies to any costumed role-playing in venues apart from the stage. Any entity that lends itself to dramatic interpretation may be taken up as a subject. Favorite sources include anime, cartoons, comic books, manga, television series, rock music performances, video games and in some cases original characters. The term is composed of the two aforementioned counterparts – costume and role play."
    input_text5 = "Support:  Linguistics is the scientific study of language. Linguistics is based on a theoretical as well as a descriptive study of language and is also interlinked with the applied fields of language studies and language learning, which entails the study of specific languages. Before the 20th century, linguistics evolved in conjunction with literary study and did not employ scientific methods.[4] Modern-day linguistics is considered a science because it entails a comprehensive, systematic, objective, and precise analysis of all aspects of language[4] – i.e., the cognitive, the social, the cultural, the psychological, the environmental, the biological, the literary, the grammatical, the paleographical, and the structural."
    input_text6 = "Support:  The scientific method was argued for by Enlightenment philosopher Francis Bacon, rose to popularity with the discoveries of Isaac Newton and his followers, and continued into later eras. In the early eighteenth century, there existed an epistemic virtue in science which has been called truth-to-nature. This ideal was practiced by Enlightenment naturalists and scientific atlas-makers, and involved active attempts to eliminate any idiosyncrasies in their representations of nature in order to create images thought best to represent \"what truly is\". Judgment and skill were deemed necessary in order to determine the \"typical\", \"characteristic\", \"ideal\", or \"average\". In practicing, truth-to-nature naturalists did not seek to depict exactly what was seen; rather, they sought a reasoned image."
    input_text7 = "Support:  Benthic locomotion is movement by animals that live on, in, or near the bottom of aquatic environments. In the sea, many animals walk over the seabed. Echinoderms primarily use their tube feet to move about. The tube feet typically have a tip shaped like a suction pad that can create a vacuum through contraction of muscles. This, along with some stickiness from the secretion of mucus, provides adhesion. Waves of tube feet contractions and relaxations move along the adherent surface and the animal moves slowly along.[11] Some sea urchins also use their spines for benthic locomotion."
    input_text8 = "Support:  "
    input_text9 = "Support:  "
    input_text10 = "Support:  "
    input_text11 = "Support:  "
    
    
    generate_QA(input_text1)
    generate_QA(input_text2)
    generate_QA(input_text3)
    generate_QA(input_text4)
    generate_QA(input_text5)
    generate_QA(input_text6)
    generate_QA(input_text7)
    
if __name__ == '__main__':
    main()
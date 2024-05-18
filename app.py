import random
from textblob import TextBlob
import spacy
from transformers import pipeline

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Load GPT-2 model
generator = pipeline('text-generation', model='gpt2')

# Define the choices for each word
first_word_choices = [
    'Once', 'Suddenly', 'Magically', 'Surprisingly', 'Quickly', 'Slowly',
    'Happily', 'Sadly', 'Curiously', 'Mysteriously', 'Excitedly', 'Strangely',
    'Joyfully', 'Bravely', 'Secretly', 'Unexpectedly', 'Calmly', 'Eagerly',
    'Lazily', 'Nervously'
]
second_word_choices = [
    'is', 'are', 'was', 'were', 'am', 'be', 'been', 'being', 'has', 'have',
    'had', 'do', 'does', 'did', 'will', 'would', 'shall', 'should', 'may',
    'might'
]
third_word_choices = [
    'cat', 'dog', 'house', 'car', 'book', 'tree', 'chair', 'table', 'water',
    'food', 'ball', 'computer', 'phone', 'bed', 'shirt', 'shoes', 'hat',
    'sun', 'moon', 'star', 'cloud', 'rain', 'snow', 'fire', 'earth', 'wind',
    'river', 'ocean', 'mountain'
]
fourth_word_choices = [
    'in', 'with', 'on', 'under', 'above', 'beside', 'between', 'before',
    'after', 'near', 'far', 'behind', 'inside', 'outside', 'through',
    'across', 'against', 'towards', 'upon', 'within', 'without', 'along',
    'around', 'beneath', 'despite', 'throughout', 'among'
]
fifth_word_choices = third_word_choices  # Reuse the third_word_choices list


def create_random_sentence():
    # Randomly select one word from each array
    first_word = random.choice(first_word_choices)
    second_word = random.choice(second_word_choices)
    third_word = random.choice(third_word_choices)
    fourth_word = random.choice(fourth_word_choices)
    fifth_word = random.choice(fifth_word_choices)

    # Construct the sentence
    sentence = f"{first_word} {second_word} {third_word} {fourth_word} {fifth_word}"

    # Correct and capitalize the sentence using TextBlob
    try:
        blob = TextBlob(sentence)
        sentence = str(blob.correct())
    except Exception as e:
        print(f"Error correcting sentence with TextBlob: {e}")

    # Capitalize the sentence
    sentence = sentence.capitalize()

    return sentence


def generate_enhanced_sentence():
    random_sentence = create_random_sentence()

    # Process the sentence with spaCy for further enhancements if needed
    doc = nlp(random_sentence)
    enhanced_sentence = ' '.join([token.text for token in doc])

    # Generate continuation using GPT-2 for more creativity
    gpt2_output = generator(
        enhanced_sentence, max_length=50, num_return_sequences=1, truncation=True, pad_token_id=50256)

    # Ensure the output ends with a full stop
    final_sentence = gpt2_output[0]['generated_text']
    final_sentence = final_sentence.split('. ')[0] + '.'

    return final_sentence


# Generate and print an enhanced sentence
enhanced_sentence = generate_enhanced_sentence()
print(enhanced_sentence)

import random
import nltk
from nltk.stem import PorterStemmer
from data import data # Assuming your data is in 'data.py'
import wikipedia


try:
    nltk.data.find("tokenizers/punkt")
except nltk.downloader.DownloadError:
    nltk.download("punkt")
    print("NLTK 'punkt' tokenizer downloaded.")

stemmer = PorterStemmer()

INTENT_RESPONSE_MAP = {
    "greetings": "responses",
    "farewells": "farewell_responses",
    "small_talk": "small_talk_responses"
}


STOP_WORDS = set([
    'what', 'is', 'the', 'of', 'a', 'an', 'are', 'do', 'does', 'did', 'who', 'where',
    'when', 'why', 'how', 'tell', 'me', 'about', 'can', 'you', 'give', 'information',
    'on', 'be', 'to', 'and', 'in', 'it', 'for', 'this', 'that', 'i', 'am', 'your', 'my' # Added a few more common ones
])

def preprocess(sentence):
    """Tokenizes and stems a sentence."""
    tokens = nltk.word_tokenize(sentence.lower())
    return [stemmer.stem(token) for token in tokens]

def clean_for_wikipedia_query(user_input):
    tokens = nltk.word_tokenize(user_input.lower())
    filtered_tokens = [
        token for token in tokens
        if token not in STOP_WORDS and token.isalnum()
    ]
    return ' '.join(filtered_tokens)

def get_wikipedia_summary(query):
    
    try:
        
        summary = wikipedia.summary(query, sentences=2, auto_suggest=True)
        return summary
    except wikipedia.exceptions.DisambiguationError as e:
        options = e.options[:3] # Show up to 3 options
        print(f"DEBUG: Wikipedia Disambiguation Error for query '{query}': Options were {e.options}") # DEBUG print
        return f"Your query is a bit ambiguous. Did you mean one of these: {', '.join(options)}? Please try to be more specific."
    except wikipedia.exceptions.PageError:
        print(f"DEBUG: Wikipedia Page Not Found Error for query: '{query}'") # DEBUG print
        return f"I couldn't find anything on Wikipedia about '{query}'. Please try another query."
    except Exception as e:
        print(f"DEBUG: General Wikipedia Error for query '{query}': {e}") # DEBUG print
        return f"An error occurred while trying to access Wikipedia: {e}"

def get_response(user_input):
    
    processed_input = preprocess(user_input)
    print(f"DEBUG: Processed User Input (for intent matching): {processed_input}")

    
    for intent_category, response_category in INTENT_RESPONSE_MAP.items():
        for pattern in data[intent_category]:
            processed_pattern = preprocess(pattern)
            if all(word in processed_input for word in processed_pattern):
                print(f"DEBUG: Matched Intent: {intent_category} with pattern: '{pattern}'") # DEBUG print
                return random.choice(data[response_category])

    wiki_query = clean_for_wikipedia_query(user_input)

    print(f"DEBUG: Cleaned Wikipedia Query: '{wiki_query}' (Original: '{user_input}')")

    if len(wiki_query) > 2: # Ensure the cleaned query is not too short
        print(f"Chatbot: Searching Wikipedia for '{user_input}' (querying as '{wiki_query}')...")
        wiki_response = get_wikipedia_summary(wiki_query)
        if wiki_response:
            return wiki_response
        else:
            return "I couldn't find information on that. Can you rephrase or ask something else?"
    else:
        print(f"DEBUG: Wikipedia query too short or empty after cleaning: '{wiki_query}'")  


    return "I am not sure how to respond to that. Could you rephrase?"

def chat():
    """Main chat loop."""
    print("Chatbot : Hello! I'm your friendly chatbot.")
    print("Chatbot : I can answer general knowledge questions using Wikipedia.")
    print("Chatbot : Type 'exit' to end the conversation.")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Chatbot: Goodbye! Have a great day!")
            break
        response = get_response(user_input)
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    chat()

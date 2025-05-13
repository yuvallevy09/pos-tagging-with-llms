# --- Imports ---
import os
from google import generativeai as genai
from pydantic import BaseModel, Field
from typing import List, Literal, Optional
from enum import Enum


# 
gemini_model = 'gemini-2.0-flash-lite'

# --- Define Pydantic Models for Structured Output ---

# --- Define the Universal Dependencies POS Tagset (17 core tags) as an enum ---
class UDPosTag(str, Enum):
    ADJ = "ADJ"     # adjective
    ADP = "ADP"     # adposition
    ADV = "ADV"     # adverb
    AUX = "AUX"     # auxiliary verb
    CCONJ = "CCONJ" # coordinating conjunction
    DET = "DET"     # determiner
    INTJ = "INTJ"   # interjection
    NOUN = "NOUN"   # noun
    NUM = "NUM"     # numeral
    PART = "PART"   # particle
    PRON = "PRON"   # pronoun
    PROPN = "PROPN" # proper noun
    PUNCT = "PUNCT" # punctuation
    SCONJ = "SCONJ" # subordinating conjunction
    SYM = "SYM"     # symbol
    VERB = "VERB"   # verb
    X = "X"         # other / unknown

# TODO Define more Pydantic models for structured output

class TokenPOS(BaseModel):
    text: str = Field(description="The token text")
    pos_tag: UDPosTag = Field(description="The Universal Dependencies POS tag")

class SentencePOS(BaseModel):
    tokens: List[TokenPOS] = Field(description="List of tokens with their POS tags")

class TaggedSentences(BaseModel):
    """Represents a list of sentences with their tagged tokens."""
    sentences: List[SentencePOS] = Field(description="A list of sentences, each containing tagged tokens.")

# --- Configure the Gemini API ---
# Get a key https://aistudio.google.com/plan_information 
# Use os.environ.get for production environments.
# For Colab/AI Studio, you might use userdata.get
# Example:
# from google.colab import userdata
# GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
# genai.configure(api_key=GOOGLE_API_KEY)

# Make sure to replace "YOUR_API_KEY" with your actual key if running locally
# and not using environment variables or userdata.
try:
    # Attempt to get API key from environment variable
    def load_env_from_ini(filename):
        with open(filename) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()

    # Load the API key
    load_env_from_ini("gemini_key.ini")
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        # Fallback or specific instruction for local setup
        # Replace with your actual key if needed, but environment variables are safer
        api_key = "YOUR_API_KEY"
        if api_key == "YOUR_API_KEY":
           print("⚠️ Warning: API key not found in environment variables. Using placeholder.")
           print("   Please set the GOOGLE_API_KEY environment variable or replace 'YOUR_API_KEY' in the code.")

    genai.configure(api_key=api_key)

except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    print("Please ensure you have a valid API key set.")
    # Depending on the environment, you might want to exit here
    # import sys
    # sys.exit(1)


# --- Function to Perform POS Tagging ---

def tag_sentences_ud(text_to_tag: str) -> Optional[TaggedSentences]:
    """
    Performs POS tagging on the input text using the Gemini API and
    returns the result structured according to the SentencePOS Pydantic model.

    Args:
        text_to_tag: The sentence or text to be tagged.

    Returns:
        A TaggedSentences object containing the tagged tokens, or None if an error occurs.
    """
    # Construct the prompt
    prompt = f"""You are a part-of-speech tagger using the Universal Dependencies (UD) tagset. Your task is to analyze the given text, split it into tokens according to UD guidelines, and assign the correct POS tag to each token.

        The UD POS tagset consists of 17 universal tags:
        1. ADJ: adjective (e.g., big, old, green)
        2. ADP: adposition (e.g., in, to, during)
        3. ADV: adverb (e.g., very, well, exactly)
        4. AUX: auxiliary verb (e.g., is, has, must)
        5. CCONJ: coordinating conjunction (e.g., and, or, but)
        6. DET: determiner (e.g., a, an, the)
        7. INTJ: interjection (e.g., oh, wow, ah)
        8. NOUN: noun (e.g., car, girl, tree)
        9. NUM: numeral (e.g., 2, two, second)
        10. PART: particle (e.g., 's in "John's", to in "going to")
        11. PRON: pronoun (e.g., I, you, she)
        12. PROPN: proper noun (e.g., John, London, IBM)
        13. PUNCT: punctuation (e.g., ., !, ?)
        14. SCONJ: subordinating conjunction (e.g., if, while, that)
        15. SYM: symbol (e.g., $, %, §)
        16. VERB: verb (e.g., run, eat, play)
        17. X: other (for words that cannot be assigned a category)

        Follow these UD segmentation guidelines:
        - Split contracted forms (e.g., "don't" → "do" [AUX] + "n't" [PART])
        - Split hyphenated compounds into separate tokens (e.g., "search-engine" → "search" [NOUN] + "-" [PUNCT] + "engine" [NOUN])
        - Treat multiword tokens as separate tokens (e.g., "because of" → "because" [SCONJ] + "of" [ADP])
        - Separate punctuation from words (e.g., "(hello)" → "(" [PUNCT] + "hello" [NOUN] + ")" [PUNCT])

        Please analyze the following text and tag each token:
        {text_to_tag}

        Return your analysis in a structured format where each sentence is split into tokens, and each token has its corresponding POS tag.
    """

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=gemini_model,
        contents=prompt,
        config={
            'response_mime_type': 'application/json',
            'response_schema': TaggedSentences,
        },
    )
    # Use the response as a JSON string.
    print(response.text)

    # Use instantiated objects.
    res: TaggedSentences = response.parsed
    return res

def batch_tag_sentences_ud(sentences: List[str], batch_size: int = 5, rate_limit_delay: float = 4.0) -> List[TaggedSentences]:
    """
    Performs POS tagging on a list of sentences in batches to respect API rate limits.
    Args:
    sentences: List of sentences to be tagged
    batch_size: Number of sentences to include in each API call
    rate_limit_delay: Delay in seconds between API calls to respect rate limits
    Returns:
    List of TaggedSentences objects
    """
    results = []
    # Process sentences in batches
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        # Join the batch into a single text with sentence separators
        batch_text = "\n\n".join(batch)
        print(f"Processing batch {i//batch_size + 1}/{(len(sentences) + batch_size - 1)//batch_size}...")
        # Process the batch
        batch_result = tag_sentences_ud(batch_text)
        if batch_result:
            results.append(batch_result)
        # Respect rate limits by adding a delay between API calls
        if i + batch_size < len(sentences):
            print(f"Waiting {rate_limit_delay} seconds to respect API rate limits...")
        time.sleep(rate_limit_delay)
    return results


# --- Example Usage ---
if __name__ == "__main__":
    # example_text = "The quick brown fox jumps over the lazy dog."
    example_text = "What if Google expanded on its search-engine (and now e-mail) wares into a full-fledged operating system?"
    # example_text = "Google Search is a web search engine developed by Google LLC."
    # example_text = "החתול המהיר קופץ מעל הכלב העצלן." # Example in Hebrew

    print(f"\nTagging text: \"{example_text}\"")

    tagged_result = tag_sentences_ud(example_text)

    if tagged_result:
        print("\n--- Tagging Results ---")
        for s in tagged_result.sentences:
            # TODO: Retrieve tokens and tags from each sentence:
            for token in s.tokens:
                tag = token.pos_tag
                token = token.text
                # Handle potential None for pos_tag if model couldn't assign one
                ctag = tag if tag is not None else "UNKNOWN"
                print(f"Token: {token:<15} {str(ctag)}")
                print("----------------------")
    else:
        print("\nFailed to get POS tagging results.")

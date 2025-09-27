# Jordan Boyd-Graber
# 2023
#
# Feature extractors to improve classification to determine if an answer is
# correct.

from collections import Counter
from math import log
import gzip
import re
from typing import Dict, List, Optional
import time

class Feature:
    """
    Base feature class.  Needs to be instantiated in params.py and then called
    by buzzer.py
    """

    def __init__(self, name):
        self.name = name

    def __call__(self, question, run, guess, guess_history, other_guesses=None):
        """

        question -- The JSON object of the original question, you can extract metadata from this such as the category

        run -- The subset of the question that the guesser made a guess on

        guess -- The guess created by the guesser

        guess_history -- Previous guesses (needs to be enabled via command line argument)

        other_guesses -- All guesses for this run
        """

        raise NotImplementedError(
            "Subclasses of Feature must implement this function")



"""
Given features (Length, Frequency)
"""
class LengthFeature(Feature):
    """
    Feature that computes how long the inputs and outputs of the QA system are.
    """

    def __call__(self, question, run, guess, guess_history, other_guesses=None):
        # How many characters long is the question?
        question_char_length = len(question.text) if hasattr(question, 'text') else len(str(question))
        yield ("question_char_length", question_char_length)

        # How many words long is the question?
        question_word_length = len(question.text.split()) if hasattr(question, 'text') else len(str(question).split())
        yield ("question_word_length", question_word_length)

        guess_length = 0

        # How many characters long is the guess?
        if guess is None or guess == "":
            yield ("guess_word_length", guess_length)
        else:
            guess_length = len(guess.split())
            yield ("guess_word_length", guess_length)


class FrequencyFeature:
    def __init__(self, name):
        from eval import normalize_answer
        self.name = name
        self.counts = Counter()
        self.normalize = normalize_answer

    def add_training(self, question_source):
        import json
        with gzip.open(question_source) as infile:
            questions = json.load(infile)
        for ii in questions:
            self.counts[self.normalize(ii["page"])] += 1

    def __call__(self, question, run, guess, guess_history, guesses):
        # We only use question, run, and guess (same as before)
        # guess_history and guesses are ignored since we don't need them

        frequency_value = log(1 + self.counts[self.normalize(guess)])
        return [("guess", frequency_value)]


class KeywordFeature:
    """
    capital words and group adjacent capital words together
    """

    def __init__(self, name):
        self.name = name
        # Common stop words
        self.stop_words = {
            'I', 'YOU', 'HE', 'SHE', 'IT', 'WE', 'THEY', 'THE', 'AND', 'OR', 'BUT', 'IN', 'ON', 'AT', 'TO', 'FOR', 'OF',
            'WITH', 'BY', 'A', 'AN', 'IS', 'ARE', 'WAS', 'WERE', 'BE', 'BEEN', 'HAVE', 'HAS', 'HAD', 'DO', 'DOES',
            'DID', 'WILL', 'WOULD', 'COULD', 'SHOULD', 'MAY', 'MIGHT', 'THIS', 'THAT', 'THESE', 'THOSE'
        }

    def extract_capital_keywords(self, text):
        if not text:
            return []

        text_str = text.text if hasattr(text, 'text') else str(text)

        # start with capital letters
        # regex capitalized words
        capital_sequences = re.findall(r'[A-Z][a-z]*(?:\s+[A-Z][a-z]*)*', text_str)

        keywords = []
        for sequence in capital_sequences:
            # filter stop words
            words = sequence.strip().split()
            filtered_words = [word for word in words if word.upper() not in self.stop_words]

            if filtered_words:
                # multiple -> join them
                keyword = ' '.join(filtered_words)
                keywords.append(keyword)

        # acronyms
        acronyms = re.findall(r'\b[A-Z]{2,}\b', text_str)
        for acronym in acronyms:
            if acronym not in self.stop_words:
                keywords.append(acronym)

        return keywords

    def extract_additional_keywords(self, text):
        if not text:
            return []

        text_str = text.text if hasattr(text, 'text') else str(text)

        keywords = []

        # quotes
        quoted_words = re.findall(r'"([^"]+)"', text_str)
        keywords.extend([word.strip() for word in quoted_words if len(word.strip()) > 1])

        # appear multiple times
        words = re.findall(r'\b[A-Za-z]{3,}\b', text_str.lower())
        word_counts = Counter(words)
        frequent_words = [word for word, count in word_counts.items()
                          if count > 1 and word.upper() not in self.stop_words]
        keywords.extend(frequent_words)

        return keywords

    def __call__(self, question, run, guess, guess_history, other_guesses=None):
        """Extract keyword features from guess."""

        if guess and guess != "":
            guess_capital_keywords = self.extract_capital_keywords(guess)
            guess_additional_keywords = self.extract_additional_keywords(guess)

            all_guess_keywords = list(set(guess_capital_keywords + guess_additional_keywords))
            yield ("guess_keyword_count", len(all_guess_keywords))
            yield ("guess_capital_keywords", len(guess_capital_keywords))
        else:
            yield ("guess_keyword_count", 0)
            yield ("guess_capital_keywords", 0)


class ConfidenceTimingFeature:
    """
    detects too early and too late guessing
    by analyzing the relationship between confidence, text position, and the quality of guesses
    """

    def __init__(self, name):
        self.name = name
        # Common ending phrases
        self.ending_phrases = [
            'for 10 points', 'for ten points', 'name this', 'identify this',
            'what is this', 'who is this', 'ftp', 'for the win'
        ]

    def calculate_text_completion_ratio(self, text, total_expected_length=500):
        # how much of the question text we've seen (0.0 to 1.0)
        if not text:
            return 0.0

        text_str = text.text if hasattr(text, 'text') else str(text)
        current_length = len(text_str)

        # Estimate based on length and ending phrases
        length_ratio = min(1.0, current_length / total_expected_length)

        # ending phrases = question is nearly complete
        ending_bonus = 0.0
        text_lower = text_str.lower()
        for phrase in self.ending_phrases:
            if phrase in text_lower:
                ending_bonus = 0.3
                break

        return min(1.0, length_ratio + ending_bonus)

    def count_specific_clues(self, text):
        if not text:
            return 0

        text_str = text.text if hasattr(text, 'text') else str(text)
        clue_count = 0

        # names
        names = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text_str)
        clue_count += len([name for name in names if len(name.split()) <= 3])  # Avoid long phrases

        # dates
        dates = re.findall(r'\b\d{4}\b|\b\d{1,2}th century\b', text_str)
        clue_count += len(dates)

        # quotes phrases
        quotes = re.findall(r'"[^"]{3,}"', text_str)
        clue_count += len(quotes)

        # numbers: years, quantities
        specific_numbers = re.findall(r'\b\d{2,4}\b', text_str)
        clue_count += len(specific_numbers)

        return clue_count

    def detect_premature_specificity(self, text, guess):
        # guess overly specific given info
        if not guess or guess == "":
            return False

        text_str = text.text if hasattr(text, 'text') else str(text)
        guess_str = str(guess)

        # if guess has multiple specific words but text is short, might be premature
        guess_words = guess_str.split()
        text_length = len(text_str)

        # might be aggressive
        if len(guess_words) >= 2 and text_length < 300:
            return True

        # very specific terms
        guess_specific_terms = len(re.findall(r'\b[A-Z][a-z]+\b', guess_str))
        text_specific_terms = len(re.findall(r'\b[A-Z][a-z]+\b', text_str))

        # guess specificity > text specificity, jumping to conclusions
        if guess_specific_terms > 0 and text_specific_terms / max(1, guess_specific_terms) < 2:
            return True

        return False

    def __call__(self, question, run, guess, guess_history, other_guesses=None):
        # identify aggressive timing

        # how complete the question appears to be
        completion_ratio = self.calculate_text_completion_ratio(question)
        yield ("text_completion_ratio", completion_ratio)

        # specific clues
        clue_count = self.count_specific_clues(question)
        yield ("available_clue_count", clue_count)

        if guess and guess != "":
            is_premature = self.detect_premature_specificity(question, guess)
            yield ("appears_premature", int(is_premature))

            guess_words = len(str(guess).split())
            clue_density = clue_count / max(1, guess_words)
            yield ("clue_to_guess_word_ratio", clue_density)

            # late indicator: high completion but simple guess
            if completion_ratio > 0.8 and guess_words == 1:
                yield ("late_simple_guess", 1)
            else:
                yield ("late_simple_guess", 0)

            aggressiveness = max(0, (2 - completion_ratio) * guess_words * 0.5)
            yield ("aggressiveness_score", aggressiveness)

            timidity = max(0, (completion_ratio - 0.7) * (4 - guess_words) * 0.5)
            yield ("timidity_score", timidity)

        else:
            yield ("appears_premature", 0)
            yield ("clue_to_guess_word_ratio", 0)
            yield ("late_simple_guess", 0)
            yield ("aggressiveness_score", 0)
            yield ("timidity_score", 0)


class AnswerSpecificityFeature:
    """
    mismatches between question type, guess specificity, and expected answer format
    identify over-specific or under-specific
    """

    def __init__(self, name):
        self.name = name

        # q type indicators
        self.person_indicators = [
            'this author', 'this writer', 'this philosopher', 'this thinker', 'this composer',
            'this scientist', 'this mathematician', 'this king', 'this emperor', 'this president',
            'this character', 'this figure', 'who is', 'who was'
        ]

        self.work_indicators = [
            'this novel', 'this work', 'this book', 'this play', 'this poem', 'this symphony',
            'this painting', 'this opera', 'what work', 'what novel', 'what book'
        ]

        self.concept_indicators = [
            'this type', 'this kind', 'this form', 'this process', 'this phenomenon',
            'this term', 'what term', 'what type', 'what kind', 'what process'
        ]

        self.place_indicators = [
            'this city', 'this country', 'this region', 'this sea', 'this mountain',
            'where is', 'what city', 'what country', 'geographic feature'
        ]

        # Common over-specification patterns
        self.title_additions = ['of france', 'of england', 'of spain', 'the great', 'i', 'ii', 'iii']
        self.unnecessary_words = ['saint', 'the', 'a', 'an']

    def detect_question_type(self, text):
        """Classify the type of question being asked"""
        if not text:
            return "unknown"

        text_str = (text.text if hasattr(text, 'text') else str(text)).lower()

        # Check for different question types
        if any(indicator in text_str for indicator in self.person_indicators):
            return "person"
        elif any(indicator in text_str for indicator in self.work_indicators):
            return "work"
        elif any(indicator in text_str for indicator in self.concept_indicators):
            return "concept"
        elif any(indicator in text_str for indicator in self.place_indicators):
            return "place"
        else:
            return "unknown"

    def analyze_guess_specificity(self, guess):
        """Analyze how specific a guess is"""
        if not guess or guess == "" or guess.lower() == "none":
            return {
                'is_none': True,
                'word_count': 0,
                'has_titles': False,
                'has_dates': False,
                'has_location': False,
                'specificity_score': 0
            }

        guess_str = str(guess).lower()
        words = guess_str.split()

        # Check for various specificity markers
        has_titles = any(title in guess_str for title in self.title_additions)
        has_dates = bool(re.search(r'\b(i{1,3}|iv|v|vi{0,3}|ix|x)\b', guess_str))  # Roman numerals
        has_location = any(loc in guess_str for loc in ['of ', 'the ', 'in ', 'from '])

        # Calculate specificity score
        specificity_score = len(words)
        if has_titles: specificity_score += 1
        if has_dates: specificity_score += 1
        if has_location: specificity_score += 0.5

        return {
            'is_none': False,
            'word_count': len(words),
            'has_titles': has_titles,
            'has_dates': has_dates,
            'has_location': has_location,
            'specificity_score': specificity_score
        }

    def detect_technical_vs_simple_mismatch(self, text, guess):
        """Detect when technical questions get simple answers or vice versa"""
        if not text or not guess or guess == "":
            return 0

        text_str = (text.text if hasattr(text, 'text') else str(text)).lower()
        guess_str = str(guess).lower()

        # Technical question indicators
        technical_terms = ['proto-indo-european', 'chomsky hierarchy', 'ablaut', 'syntax',
                           'complementizer', 'derivation', 'linguistics', 'copula']

        # Simple answer indicators
        simple_answers = ['none', 'zero', 'null', 'empty']

        has_technical_terms = sum(1 for term in technical_terms if term in text_str)
        is_simple_answer = any(simple in guess_str for simple in simple_answers)

        # Mismatch: highly technical question with overly simple answer
        if has_technical_terms >= 2 and is_simple_answer:
            return 1
        return 0

    def calculate_question_answer_alignment(self, text, guess):
        """Calculate how well the guess aligns with what the question is asking for"""
        if not text or not guess or guess == "":
            return 0

        question_type = self.detect_question_type(text)
        guess_analysis = self.analyze_guess_specificity(guess)

        # question has lots of specific details but we guess "None"
        text_str = text.text if hasattr(text, 'text') else str(text)
        specific_clues = len(re.findall(r'\b[A-Z][a-z]+\b', text_str))

        if specific_clues > 5 and guess_analysis['is_none']:
            return -1.0  #  penalty for under-confidence with good clues

        return 0

    def __call__(self, question, run, guess, guess_history, other_guesses=None):
        """Generate features for answer specificity analysis"""

        # Basic question analysis
        question_type = self.detect_question_type(question)
        yield ("question_type_person", 1 if question_type == "person" else 0)
        yield ("question_type_work", 1 if question_type == "work" else 0)
        yield ("question_type_concept", 1 if question_type == "concept" else 0)
        yield ("question_type_place", 1 if question_type == "place" else 0)

        if guess and guess != "":
            # Analyze guess specificity
            guess_analysis = self.analyze_guess_specificity(guess)

            yield ("guess_is_none", 1 if guess_analysis['is_none'] else 0)
            yield ("guess_specificity_score", guess_analysis['specificity_score'])
            yield ("guess_has_location_qualifier", 1 if guess_analysis['has_location'] else 0)
            yield ("guess_has_title_qualifier", 1 if guess_analysis['has_titles'] else 0)


            # Alignment score
            alignment = self.calculate_question_answer_alignment(question, guess)
            yield ("question_answer_alignment", alignment)

            # Over-specification detection (like "Louis XIII of France" vs "Louis_XIII")
            if guess_analysis['word_count'] >= 3 and (
                    guess_analysis['has_titles'] or guess_analysis['has_location']
            ):
                yield ("potentially_over_specific", 1)
            else:
                yield ("potentially_over_specific", 0)

            # Under-confidence detection (giving up with "None" when clues exist)
            text_str = question.text if hasattr(question, 'text') else str(question)
            clue_count = len(re.findall(r'\b[A-Z][a-z]+\b', text_str))

            if guess_analysis['is_none'] and clue_count > 3:
                yield ("potentially_under_confident", 1)
            else:
                yield ("potentially_under_confident", 0)

        else:
            # No guess provided
            yield ("guess_is_none", 1)
            yield ("guess_specificity_score", 0)
            yield ("guess_has_location_qualifier", 0)
            yield ("guess_has_title_qualifier", 0)
            yield ("question_answer_alignment", 0)
            yield ("potentially_over_specific", 0)
            yield ("potentially_under_confident", 1)  # No guess is under-confident


class FeatureContext:
    """Context object containing external data sources for feature generation"""
    wikipedia_api: Optional[object] = None
    temporal_data: Optional[Dict] = None
    question_metadata: Optional[Dict] = None
    historical_performance: Optional[List] = None
    external_knowledge: Optional[Dict] = None

class EnhancedFeature:
    """Base class for features that can use external data"""

    def __init__(self, name):
        self.name = name

    def __call__(self, question, run, guess, guess_history, other_guesses=None, context=None):
        """call method with context parameter"""
        # Default implementation - subclasses should override
        return []


class TemporalDynamicsFeature(EnhancedFeature):
    """captures temporal info"""

    def __init__(self, name):
        super().__init__(name)
        self.guess_history = []
        self.timing_history = []

    def update_history(self, guess, timestamp, confidence=None):
        # internal history tracking
        self.guess_history.append({
            'guess': guess,
            'timestamp': timestamp,
            'confidence': confidence
        })

        # last 100 entries
        if len(self.guess_history) > 100:
            self.guess_history = self.guess_history[-100:]

    def __call__(self, question, run, guess, guess_history, other_guesses=None, context=None):
        current_time = time.time()

        if len(self.guess_history) > 0:
            time_since_last = current_time - self.guess_history[-1]['timestamp']
            yield ("time_since_last_guess", time_since_last)

            recent_guesses = [g for g in self.guess_history
                              if current_time - g['timestamp'] < 60]  # Last minute
            yield ("recent_guess_frequency", len(recent_guesses))

            if len(self.guess_history) >= 3:
                recent_confidences = [g.get('confidence', 0) for g in self.guess_history[-3:]]
                if all(c is not None for c in recent_confidences):
                    confidence_trend = recent_confidences[-1] - recent_confidences[0]
                    yield ("confidence_trend", confidence_trend)

        # update history
        self.update_history(guess, current_time,
                            context.temporal_data.get('confidence') if context and context.temporal_data else None)



class GuessBlankFeature(Feature):
    """
    Is guess blank?
    """

    def __call__(self, question, run, guess, guess_history, other_guesses=None):
        yield ('true', len(guess) == 0)


class GuessCapitalsFeature(Feature):
    """
    Capital letters in guess
    """

    def __call__(self, question, run, guess, guess_history, other_guesses=None):
        yield ('true', log(sum(i.isupper() for i in guess) + 1))


if __name__ == "__main__":
    """

    Script to write out features for inspection or for data for the 470
    logistic regression homework.

    """
    import argparse

    from params import add_general_params, add_question_params, \
        add_buzzer_params, add_guesser_params, setup_logging, \
        load_guesser, load_questions, load_buzzer

    parser = argparse.ArgumentParser()
    parser.add_argument('--json_guess_output', type=str)
    add_general_params(parser)
    add_guesser_params(parser)
    add_buzzer_params(parser)
    add_question_params(parser)

    flags = parser.parse_args()

    setup_logging(flags)

    guesser = load_guesser(flags)
    buzzer = load_buzzer(flags)
    questions = load_questions(flags)

    buzzer.add_data(questions)
    buzzer.build_features(flags.buzzer_history_length,
                          flags.buzzer_history_depth)

    vocab = buzzer.write_json(flags.json_guess_output)
    with open("data/small_guess.vocab", 'w') as outfile:
        for ii in vocab:
            outfile.write("%s\n" % ii)
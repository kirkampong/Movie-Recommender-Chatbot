#!/usr/bin/env python
# -*- coding: utf-8 -*-

# PA6, CS124, Stanford, Winter 2016
# v.1.0.2
# Original Python code by Ignacio Cases (@cases)
# Ported to Java by Raghav Gupta (@rgupta93) and Jennifer Lu (@jenylu)
######################################################################
import csv
import math
import re
from PorterStemmer import PorterStemmer
import numpy as np
import random

from movielens import ratings
from random import randint


class Chatbot:
    """Simple class to implement the chatbot for PA 6."""

    #############################################################################
    # `moviebot` is the default chatbot. Change it to your chatbot's name       #
    #############################################################################
    def __init__(self, is_turbo=False):
        self.name = 'moviebot'
        self.is_turbo = is_turbo
        self.p = PorterStemmer()
        self.read_data()
        self.threshold = 0
        self.user_scores = np.zeros((len(self.ratings), 1))
        self.seenMovies = []
        self.num_data_points = 0
        self.data_asks = 3

        self.disambiguate_flag = False
        self.disambiguate_list = []
        self.ambiguous_title = ''
        self.ambiguous_input = ''

        self.binarized = []
        self.strong_flag = False
        self.sad_flag = False
        self.scared_flag = False
        self.trained = False
        self.binarize()
        self.p = PorterStemmer()

        self.NEGATORS = set(['not', 'neither', 'never', 'no', 'cannot', 'wont', 'cant'])
        self.negRegex = re.compile(r"[a-z]+n't$")
        self.EMPHASIZERS = set(['very', 'extremely', 'so', 'really', 'incredibly', 'remarkably',
                                'distinctly', 'insanely', 'hella', 'fucking', 'super'])
        self.PUNCTUATIONS = set([';', '.', ',', '"', '(', ')', '&', ':', '?', '[', ']'])
        self.STRONG_WORDS = set(['love', 'adore', 'amazing', 'best', 'wonderful', 'fantastic',
                                 'perfect', 'best', 'riveting', 'NEGATIVESSS-->', 'hate', 'terrible', 'despise',
                                 'pathetic', 'atricious', 'worst', 'abysmal', 'disaster', 'shitshow', 'boring'])
        self.SCARY_WORDS = set(['scary', 'scared', 'frightening', 'frightened', 'terrifying',
                                'spooky', 'nightmare', ])
        self.SAD_WORDS = set(['sad', 'tears', 'cry', 'cried', 'sorrow', 'grief', 'lonely',
                              'depressing', 'depressed', 'sob', 'sobbed', 'morose', 'melancholy', 'somber',
                              'weep', 'weeping', 'wept'])

    #############################################################################
    # 1. WARM UP REPL
    #############################################################################

    def greeting(self):
        """chatbot greeting message"""
        #############################################################################
        # TODO: Write a short greeting message                                      #
        #############################################################################

        greeting_message = 'How can I help you?'

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

        return greeting_message

    def goodbye(self):
        """chatbot goodbye message"""
        #############################################################################
        # TODO: Write a short farewell message                                      #
        #############################################################################

        goodbye_message = 'Have a nice day!'

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

        return goodbye_message

    #############################################################################
    # 2. Modules 2 and 3: extraction and transformation                         #
    #############################################################################
    def reset(self):
        self.user_scores = np.zeros((len(self.ratings), 1))
        self.seenMovies = []
        self.num_data_points = 0

    def fittingResponse(self, key):

        too_many_quotes = [
            "Sorry I get confused by a lot of quotations marks and sometimes I trouble thinking about more than one movie at a time. Could you make sure to reference just one movie?"]
        too_many_quotes.append(
            "Your quotation marks have left me a bit confused. Could you make it slightly clearer which movie in particular you are talking about?")
        too_many_quotes.append(
            "I have trouble making a good recommendation wwith one movie at a time. Could you double-check your quotation marks to make it easier for me to understand the movie you're telling me?")

        noquotes = ["Can wait to talk about that until we're done talking about movies? Tell me about a movie you saw!"]
        noquotes.append(
            "I'd like to talk about movies for a bit longer. How about you tell me about a movie you saw in the past five years?")
        noquotes.append("Sorry, did you talk about a movie? If you did I couldn't tell.")
        noquotes.append(
            "I'm not sure if you're actually talking about a movie or not, but if you're not let's try to get back to movies. I'd really like to give you some movie recommendations. Tell me how you felt about a movie!")
        noquotes.append(
            "Hmm. I can't tell if you're talking about a movie or not. Could you tell me about a movie you saw?")

        multipleVersions = [
            "Looks like there are multiple versions or releases about this movie. Use one of these names in your review: \n"]
        multipleVersions.append(
            "Are you talking about one of these movies? Use the name below to help me not get confused: \n")
        multipleVersions.append(
            "Looks like you could mean more than one movie. If your movie is in the list below, feel free to tell me about it again but with the listed name. \n")

        mispelling = [
            "Hey, I think there might have been a mispelling. Try using one of these names when telling me about that movie: \n"]
        mispelling.append(
            "I'm not positive, but I think you may have misspelled something. Did you mean one of these movies? \n")
        mispelling.append(
            "Hey I can't recognize that but these look really similar. If you meant one of these use the right name moving forward! \n")

        haventheard = ["Hm. I haven't heard of %s. Tell me about another movie."]
        haventheard.append(
            "Well I haven't seen %s yet, but maybe I should! Tell me about another movie so I can make a recommendation.")
        haventheard.append(
            "Well, it'll be hard for me to give a recommendation about %s since I haven't seen it yet. Let's try another movie!")

        alreadymentioned = ["I think you already told me about %s. Tell me about a movie you haven't mentioned yet."]
        alreadymentioned.append(
            "Hm, I think you already said %s. I want to hear about a movie you haven't mentioned yet!")
        alreadymentioned.append(
            "I think you already said %s. Tell me about some other movies so I can give you a good recommendation!")

        liked = ["Great to hear you liked %s. "]
        liked.append("Nice, I liked %s too! ")
        liked.append("Nice, I liked %s too! Tell me about another movie!")
        liked.append("Wonderful. I'll try to keep your preference for %s when recommending movies. ")
        liked.append("Yep %s was a cool movie. !")
        liked.append("Great, sounds like %s was your type of movie. ")
        liked.append("I'm glad to hear you liked %s! ")
        liked.append("I'm glad to hear you liked %s! ")
        liked.append("It's good to know that %s is your type of movie. ")
        liked.append("It's good to know that %s is your type of movie. ")
        liked.append("Sounds like you're a fan of movies like %s. That's wonderful. ")

        stronglike = ["Wow, you really loved %s. Tell me about another movie you saw!"]
        stronglike.append("I can tell you really liked %s! That's great! How about another movie?")
        stronglike.append("Sounds like %s was a really awesome experience for you.")
        stronglike.append("Hey, you really loved %s!. ")

        unsure = [
            "Hm. I'm not sure whether or not you liked %s. Try telling me about %s again, but with a bit more detail."]
        unsure.append(
            "It's hard for me to tell how you felt about %s. If you tell me again with a bit more detail, I'm sure I'll understand you!")
        unsure.append(
            "Wait. Did you like %s or not like %s? Tell me again with a bit more detail and I'll have a better idea what to recommend to you.")
        unsure.append(
            "I'm confused. Did you like %s? If you use a bit more detail I'll understand whether or not you liked %s.")

        disliked = ["Wow, sounds like you didn't enjoy %s. "]
        disliked.append(
            "I see that %s is not the type of movie you like. I'll try to recommend things with that in mind.")
        disliked.append("Gotcha. Sounds like you had a bad time with %s. ")
        disliked.append("You're definitely not alone in disliking %s. ")
        disliked.append("Seems like %s was not your thing. Tell me about another movie. ")
        disliked.append("I didn't like %s when I saw it either, though that was a few years ago. ")
        disliked.append("Okay, I understand that %s was definitely not for you. ")

        strongdis = ["Wow, you really didn't enjoy %s. "]
        strongdis.append("Sounds like %s was a really bad experience for you. ")
        strongdis.append("Wow, you must've really hated %s. ")
        strongdis.append(
            "Sounds like %s is definitely not what you're looking for. I'll keep that in mind for my recommendations.")

        sad = ["Sounds like a real tear-jerker! :( "]
        sad.append("Awww that's really sad! ")
        sad.append("Jeez I definitely would have cried! ")
        sad.append("That's super sad! ")
        sad.append("Sounds like a heartbreaker. That's a lot! ")
        sad.append("Wow that's like extra sad. ")
        sad.append("Sounds like it really pulls at the heartstrings :'( ")

        scared = ["Haha, sounds like that movie spooked you out. "]
        scared.append("Sounds like a terrifying experience! ")
        scared.append("Sounds frightening! ")
        scared.append("That seems incredibly scary! ")
        scared.append("You must have been freaked out! ")
        scared.append("That's pretty scary! ")

        movieask = ["Can you tell me about another movie?"]
        movieask.append("Tell me about another movie you saw?")
        movieask.append("Tell me about another movie you saw?")
        movieask.append("How about another movie?")
        movieask.append("Tell me just a few more movies. I'm almost ready to make my recommendation!")
        movieask.append("Let's hear about a different movie!")
        movieask.append("Tell me about a different movie!")
        movieask.append("Let's hear how you felt about another movie.")
        movieask.append("I'd like to hear how you felt about another movie you saw.")
        movieask.append("How about another movie you saw?")
        movieask.append("How did you feel about another movie you saw?")

        ret = ""
        if key == "toomanyquotes":
            ind = random.randint(0, len(too_many_quotes) - 1)
            return too_many_quotes[ind]

        if key == "noquotes":
            ind = random.randint(0, len(noquotes) - 1)
            return noquotes[ind]

        if key == "multipleversions":
            ind = random.randint(0, len(multipleVersions) - 1)
            return multipleVersions[ind]

        if key == "haventheard":
            ind = random.randint(0, len(haventheard) - 1)
            return haventheard[ind]

        if key == "mispelling":
            ind = random.randint(0, len(mispelling) - 1)
            return mispelling[ind]

        if key == "alreadymentioned":
            ind = random.randint(0, len(alreadymentioned) - 1)
            return alreadymentioned[ind]

        if key == "liked":
            ind = random.randint(0, len(liked) - 1)
            ret = liked[ind]

        if key == "stronglike":
            ind = random.randint(0, len(strongliked) - 1)
            ret = stronglike[ind]

        if key == "unsure":
            ind = random.randint(0, len(unsure) - 1)
            return unsure[ind]

        if key == "disliked":
            ind = random.randint(0, len(disliked) - 1)
            ret = ret + disliked[ind]

        if key == "strongdis":
            ind = random.randint(0, len(strongdis) - 1)
            ret = ret + strongdis[ind]

        if key == "sad":
            ind = random.randint(0, len(sad) - 1)
            return ret + sad[ind]

        if key == "scared":
            ind = random.randint(0, len(scared) - 1)
            return ret + scared[ind]

        if self.num_data_points < self.data_asks:
            ind = random.randint(0, len(movieask) - 1)
            ret = ret + movieask[ind]

        return ret

    def process(self, input):
        """Takes the input string from the REPL and call delegated functions
        that
          1) extract the relevant information and
          2) transform the information into a response to the user
        """
        #############################################################################
        # TODO: Implement the extraction and transformation in this method, possibly#
        # calling other functions. Although modular code is not graded, it is       #
        # highly recommended                                                        #
        #############################################################################
        if not self.trained:
            if input.lower().strip() == "sgdtrain":
                self.sgdLatentFactors()
                self.trained = True
                return "Okay, let's talk about some movies! Tell me about a movie you saw."
            elif input.lower().strip() == "alstrain":
                self.alsLatentFactors()
                self.trained = True
                return "Tell me about some movies you saw. Then I will give you some recommendations on what to see next!"
            else:
                self.ratings = self.binarized
                self.trained = True
        if self.is_turbo == True:
            # response = 'processed %s in creative mode !' % input
            if self.num_data_points < self.data_asks - 1:
                return self.gather_movie_data(input)
            else:
                ret = self.gather_movie_data(input)
                if(self.num_data_points == self.data_asks):
                    output = self.recommend()
                    self.reset()
                    return ret + "You know what, I think you would enjoy these movies: " + output

        else:
            # response = 'processed %s in starter mode' % input
            if self.num_data_points < self.data_asks - 1:
                return self.gather_movie_data(input)
            else:
                ret = self.gather_movie_data(input)
                if (self.num_data_points == self.data_asks):
                    output = self.recommend()
                    self.reset()
                    return ret + "You know what, I think you would enjoy these movies: " + output

        return response

    # Get sentiment about a movie, if sentiment is positive, set value of self.user_vector at the
    # index corresponsing to the movie to 1, if negative sentiment, set to -1.
    # TODO: Case where we are uncertain of sentiment? Value of self.threshold?
    def gather_movie_data(self, input):

        if self.disambiguate_flag:
            input = input.lower()
            self.disambiguate_flag = False
            match = self.closest_match(input, self.ambiguous_title, self.disambiguate_list)
            correct_movie = self.titles[match][0]
            # print "correct movie: " + correct_movie
            # print "Oh so you meant %s, Gotcha!" %correct_movie
            # print "amig_title: " + self.ambiguous_title
            new_input = re.sub(self.ambiguous_title, correct_movie, self.ambiguous_input)
            # print "new input: " + new_input
            return self.gather_movie_data(new_input)

        emotion = ''

        num_quotes = self.count_quotes(input)
        if num_quotes == 0:
            return self.fittingResponse("noquotes")
        elif num_quotes != 2:
            return self.fittingResponse("toomanyquotes")

        movie_title = self.get_title(input)[0]
        movie_index = self.get_movie_index(movie_title)  # gets index of movie in titles

        if type(movie_index) is set:
            answer = ''
            for index in movie_index:
                answer += self.titles[index][0] + '\n'
            return self.fittingResponse("mispelling") + answer

        if type(movie_index) is list:
            input = input.lower()
            answer = ''
            for index in movie_index:
                answer += self.titles[index][0] + '\n'
            # return self.fittingResponse("multipleversions") + answer
            self.disambiguate_flag = True
            self.disambiguate_list = movie_index
            self.ambiguous_input = input
            self.ambiguous_title = movie_title
            return "Which of these you talking about?: \n" + answer

        if movie_index == -1:
            ret = self.fittingResponse("haventheard") % movie_title
            return ret + emotion

        if movie_index in self.seenMovies:
            return self.fittingResponse("alreadymentioned") % movie_title

        score = self.get_sentiment(input)

        if self.sad_flag:
            emotion += self.fittingResponse("sad")
            self.sad_flag = False
        if self.scared_flag:
            emotion += self.fittingResponse("scared")
            self.scared_flag = False

        if score >= self.threshold + 0.2:
            self.num_data_points += 1
            self.updateScores(movie_index, 1)
            self.seenMovies.append(movie_index)
            ret = self.fittingResponse("liked") % movie_title
            if self.strong_flag:
                ret = self.fittingResponse("stronglike") % movie_title
            self.strong_flag = False
            return emotion + " " + ret
        elif score <= self.threshold - 0.2:
            self.num_data_points += 1
            self.updateScores(movie_index, -1)
            self.seenMovies.append(movie_index)
            ret = self.fittingResponse("disliked") % movie_title
            if self.strong_flag:
                ret = self.fittingResponse("strongdis") % movie_title
            self.strong_flag = False
            return emotion + " " + ret
        else:
            return self.fittingResponse("unsure") % movie_title

    def closest_match(self, reply, movie_title, movie_index):
        reply = reply.lower()
        x = len(movie_title) - 1
        closest_index = -1
        max_overlap = 0
        for index in movie_index:
            movie = self.titles[index][0].lower()
            movie = movie[x:]
            curr_overlap = self.get_overlap(reply, movie)
            if len(curr_overlap) > max_overlap:
                max_overlap = len(curr_overlap)
                closest_index = index

        return closest_index

    def get_overlap(self, s1, s2):
        s1 = s1.lower()
        s2 = s2.lower()
        s = difflib.SequenceMatcher(None, s1, s2)
        pos_a, pos_b, size = s.find_longest_match(0, len(s1), 0, len(s2))
        ret = s1[pos_a:pos_a + size]
        return ret

    def get_movie_index(self, movie_title):
        if self.is_turbo: movie_title = movie_title.lower()
        indices = []
        spell_error_candidates = set()

        if self.is_turbo:

            for i, title in enumerate(self.titles):
                m = title[0]  # eg: m = 'Toy Story (1995)'
                m = m.lower()

                # Note: check '.' after \w in the regexes

                # Year/Series Disambiguation(CREATIVE):----------------------------
                m2 = re.findall(r"(^[ a-zA-Z:_]+)(?: \d)? \(\d{4}\)",
                                m)  # disambiguate by years eg: Scream (1996) vs Scream 2 (1997)
                m3 = re.findall(r"(^[ \w:,.+-]+) (?:I+) \(\d{4}\)",
                                m)  # disamiguatie by series  eg: Clerks (1994) vs Clerks II (2006)
                m4 = re.findall(r"(^[ \w.:,+-]+): (?:[ \w.:,\)\(-]+)",
                                m)  # disamiguate by series eg: Major League: Back to the Minors (1998) vs Major League (1989)
                m5 = re.findall(r"(^[ \w:,.+-]+) and the [ \w:,.+\.'\)\(-]+ \(\d{4}\)",
                                m)  # dismbiguate by 'x' and the ... eg: Harry Potter and the Chamber of Secrets (2002) vs Harry Potter and the Prisoner of Azkaban (2004)

                # -----------------------------------------------------------------

                articles = re.findall(r"(^[ \w.+-:]+), (the|a|an|la|les|il|le) (\(\d{4}\))", m)
                if len(articles) > 0:
                    with_article = articles[0][1] + " " + articles[0][0] + " " + articles[0][2]
                    without_article = articles[0][0] + " " + articles[0][2]

                if movie_title == m:
                    return i

                elif len(articles) > 0 and (movie_title == without_article or movie_title == with_article):
                    return i

                # else:
                elif len(m2) > 0 and movie_title == m2[0]:
                    indices.append(i)
                elif len(m3) > 0 and movie_title == m3[0]:
                    indices.append(i)
                elif len(m4) > 0 and movie_title == m4[0]:
                    indices.append(i)
                elif len(m5) > 0 and movie_title == m5[0]:
                    indices.append(i)

                elif self.levenshtein_dist(movie_title, m) <= 2:  # Consider titles up to edit Distance 2 away
                    spell_error_candidates.add(i)



        else:
            for i, title in enumerate(self.titles):
                m = title[0]  # eg: m = 'Toy Story (1995)'
                # m = m.lower()
                articles = re.findall(r"(^[ \w.+-:]+), (The|A|An|La|Les|Il|Le) (\(\d{4}\))", m)
                if len(articles) > 0:
                    with_article = articles[0][1] + " " + articles[0][0] + " " + articles[0][2]
                    without_article = articles[0][0] + " " + articles[0][2]

                if movie_title == m:
                    return i

                elif len(articles) > 0 and (movie_title == without_article or movie_title == with_article):
                    return i

        if len(indices) == 1:
            return indices[0]
        elif len(indices) > 1:
            return indices
        elif len(spell_error_candidates) > 0:
            return spell_error_candidates

        return -1

    def levenshtein_dist(self, word1, word2):
        if len(word1) < len(word2):
            return self.levenshtein_dist(word2, word1)

        # len(s1) >= len(s2)
        if len(word2) == 0:
            return len(word1)

        prev_row = range(len(word2) + 1)
        for i, c1 in enumerate(word1):
            curr_row = [i + 1]
            for j, c2 in enumerate(word2):
                insertions = prev_row[j + 1] + 1
                deletions = curr_row[j] + 1
                substitutions = prev_row[j] + (c1 != c2)
                curr_row.append(min(insertions, deletions, substitutions))
            prev_row = curr_row

        return prev_row[-1]

    # Verifies Theres only one phrase in quotation marks, maybe could do other verification on input
    def count_quotes(self, sentence):
        quotation_count = 0
        for i in range(0, len(sentence)):
            if sentence[i] == '"': quotation_count += 1
        return quotation_count

    # Uses a regex tot extract the phrase in quotation marks i.e. the movie title
    def get_title(self, sentence):
        if self.is_turbo:
            return re.findall(r'"([^"]*)"', sentence.lower())
        else:
            return re.findall(r'"([^"]*)"', sentence)

    # Returns the average sentiment score of words in sentence which are not in quotation marks
    def get_sentiment(self, sentence):
        scores = []
        in_quotes = False
        negFlag = False
        words = re.findall(r"[\w']+|[.,!?;\"]",
                           sentence)  # extracts each word, but parses punctuations as entities as well
        for i, w in enumerate(words):
            # Emotion-Recognition(CREATIVE)--------------------
            if w in self.SAD_WORDS:
                self.sad_flag = True
            if w in self.SCARY_WORDS:
                self.scared_flag = True
            # -------------------------------------------------

            # Fine-grained sentiment(CREATIVE):------------------------------------------
            mult = 1  # weights score of words
            if i > 0 and words[i - 1] in self.EMPHASIZERS: mult += 1  # preceded by emphasizer |
            if w in self.STRONG_WORDS:
                mult += 1  # word is strongly sentimented            |
                self.strong_flag = True
            # ---------------------------------------------------------------------------
            w = w.lower()
            if w == '"':
                if in_quotes:
                    in_quotes = False
                else:
                    in_quotes = True
            if self.is_negative(w): negFlag = True  # negate sentiment of words b/n negator and punctuation
            if w in self.PUNCTUATIONS: negFlag = False
            w = self.p.stem(w)
            if not in_quotes and w in self.sentiment:
                if self.sentiment[w] == 'pos':
                    if negFlag == False:
                        scores.append(1 * mult)
                    elif negFlag == True:
                        scores.append(-1 * mult)
                elif self.sentiment[w] == 'neg':
                    if negFlag == False:
                        scores.append(-1 * mult)
                    elif negFlag == True:
                        scores.append(1 * mult)
        if len(scores) == 0: return 0
        return sum(scores) / float(len(scores))

    # Returns true if a word is a negator, which negates sentiment of following words
    def is_negative(self, word):
        if self.negRegex.search(word):  # takes care of 'nt
            return True
        return word in self.NEGATORS

    # 3. Movie Recommendation helper functions                                  #
    #############################################################################

    def read_data(self):
        """Reads the ratings matrix from file"""
        # This matrix has the following shape: num_movies x num_users
        # The values stored in each row i and column j is the rating for
        # movie i by user j
        self.titles, self.ratings = ratings()
        reader = csv.reader(open('data/sentiment.txt', 'rb'))
        self.sentiment = dict(reader)
        stemmedSentiment = {}
        for key in self.sentiment:
            stemmed = self.p.stem(key)
            stemmedSentiment[stemmed] = self.sentiment[key]
        self.sentiment = stemmedSentiment

    def binarize(self):
        """Modifies the ratings matrix to make all of the ratings binary"""

        # Computes the average of all nonzero ratings. Binarizes the ratings matrix by setting a rating
        # to 0 if the rating is zero (user did not rate the movie), 1 if the rating is greater than the average,
        # and -1 if the rating is less than the average

        # we could also try using the median rating and see if that does better (for future)
        averageReview = 3.54360825567
        self.binarized = np.copy(self.ratings)
        self.binarized[np.where(self.binarized > averageReview)] = -2
        self.binarized[np.where(self.binarized >= .5)] = -1
        self.binarized[np.where(self.binarized == -2)] = 1

    def sgdLatentFactors(self):
        computingErrors = False
        NUM_ITERS = 200
        Eta = .001
        Lambda = .01
        k = 30

        P_T = math.sqrt(5.0 / k) * np.random.random_sample((k, len(self.ratings[0])))
        Q = math.sqrt(5.0 / k) * np.random.random_sample((len(self.ratings), k))
        rowSum = np.sum(self.ratings, axis=1)
        zeroRows = np.nonzero(rowSum)
        for r in zeroRows:
            Q[r, :] = np.zeros((1, k))

        nonzeroIndices = []
        for user in range(0, len(self.ratings[0])):
            for movie in range(0, len(self.ratings)):
                rating = self.ratings[movie, user]
                if rating > 0:
                    nonzeroIndices.append((movie, user))
        for j in range(0, NUM_ITERS):
            for movie, user in nonzeroIndices:
                rating = self.ratings[movie, user]
                p_u = P_T[:, user]
                q_i = Q[movie, :]

                eps_iu = 2 * (rating - q_i.dot(p_u))

                Q[movie, :] = q_i + Eta * (eps_iu * np.transpose(p_u) - 2 * Lambda * q_i)
                P_T[:, user] = p_u + Eta * (eps_iu * np.transpose(q_i) - 2 * Lambda * p_u)
            print "Iteration: " + str(j)
            if computingErrors:
                E = 0
                for movie, user in nonzeroIndices:
                    rating = self.ratings[movie, user]
                    p_u = P_T[:, user]
                    q_i = Q[movie, :]
                    ratingEstimate = q_i.dot(p_u)

                    E += (rating - ratingEstimate) ** 2
                sumsq = np.sum(np.sum(np.square(P_T))) + np.sum(np.sum(np.square(Q)))
                E += Lambda * sumsq
                print E

        M = Q.dot(P_T)
        print np.max(self.ratings)
        self.ratings = M + self.ratings
        print np.max(M)
        # self.ratings = M

        averageMovieRatings = np.sum(self.ratings, axis=1) / (len(self.ratings[0]))
        averageUserRatings = np.sum(self.ratings, axis=0) / (len(self.ratings))

        if True:
            normalizationMatrix = 1 * (np.transpose(np.tile(averageMovieRatings, (len(self.ratings[0]), 1))))

        self.ratings = np.array(self.ratings) - normalizationMatrix
        self.ratings[np.where(self.ratings <= -.95)] = -1
        self.ratings[np.where(self.ratings >= .95)] = -2
        self.ratings[np.where(self.ratings > -.95)] = 0
        self.ratings[np.where(self.ratings == -2)] = 1

    def alsLatentFactors(self):
        computingErrors = True
        NUM_ITERS = 30
        Lambda = .01
        k = 30
        alpha = 1

        nonzeroIndices = []
        for user in range(0, len(self.ratings[0])):
            for movie in range(0, len(self.ratings)):
                rating = self.ratings[movie, user]
                if rating > 0:
                    nonzeroIndices.append((movie, user))

        P_T = math.sqrt(5.0 / k) * np.random.random_sample((k, len(self.ratings[0])))
        Q = math.sqrt(5.0 / k) * np.random.random_sample((len(self.ratings), k))
        W = np.zeros((len(self.ratings), len(self.ratings[0])))
        W[np.where(self.ratings > 0)] = 1
        for j in range(0, NUM_ITERS):
            Q = np.transpose(
                np.linalg.lstsq(P_T.dot(P_T.T) + Lambda * np.eye(k), P_T.dot(np.transpose(self.ratings)))[0])
            P_T = np.linalg.lstsq(np.transpose(Q).dot(Q) + Lambda * np.eye(k), np.transpose(Q).dot(self.ratings))[0]

            print "Iteration: " + str(j)
            if computingErrors:
                E = 0
                for movie, user in nonzeroIndices:
                    rating = self.ratings[movie, user]
                    p_u = P_T[:, user]
                    q_i = Q[movie, :]
                    ratingEstimate = q_i.dot(p_u)

                    E += (rating - ratingEstimate) ** 2
                sumsq = np.sum(np.sum(np.square(P_T))) + np.sum(np.sum(np.square(Q)))
                E += Lambda * sumsq
                print E

        M = Q.dot(P_T)
        print np.max(self.ratings)
        self.ratings = M + alpha * self.ratings
        print np.max(M)
        # self.ratings = M

        averageMovieRatings = np.sum(self.ratings, axis=1) / (len(self.ratings[0]))
        averageUserRatings = np.sum(self.ratings, axis=0) / (len(self.ratings))

        if True:
            normalizationMatrix = 1 * (np.transpose(np.tile(averageMovieRatings, (len(self.ratings[0]), 1))))

        self.ratings = np.array(self.ratings) - normalizationMatrix
        self.ratings[np.where(self.ratings <= -.95)] = -1
        self.ratings[np.where(self.ratings >= .95)] = -2
        self.ratings[np.where(self.ratings > -.95)] = 0
        self.ratings[np.where(self.ratings == -2)] = 1

    def updateScores(self, movie_Index, review):
        # Updates the user's scores based on their review of movie_Index
        similarities = []
        queryMovie = np.transpose(np.array(self.ratings[movie_Index]))
        if review == 1:
            dotproducts = self.ratings.dot(queryMovie)
            norms = np.linalg.norm(self.ratings, axis=1)
            norms[np.where(norms < .5)] = 10 ** 8  ##sets zero norms to infinity
            scoreUpdate = dotproducts / norms
            self.user_scores += scoreUpdate.reshape((len(self.ratings), 1))
        elif review == -1:
            dotproducts = self.ratings.dot(queryMovie)
            norms = np.linalg.norm(self.ratings, axis=1)
            norms[np.where(norms < .5)] = 10 ** 8  ##sets zero norms to infinity
            scoreUpdate = dotproducts / norms
            self.user_scores -= scoreUpdate.reshape((len(self.ratings), 1))

    def recommend(self):
        """Generates a list of movies based on the input vector u using
        collaborative filtering"""
        # TODO: Implement a recommendation function that takes a user vector u
        # and outputs a list of movies recommended by the chatbot

        if False:
            reviewed = np.extract(u > 0, u)
            numMovies = len(self.ratings)
            predictedRatings = []
            for movie in range(0, numMovies):
                if (u[movie] == 0):
                    predict = 0
                    for reviewedMovie in reviewed:
                        usersReview = u[reviewedMovie]
                        similarity = self.distance(self.ratings[movie], self.ratings[reviewedMovie])
                        predict += similarity * usersReview
                    predictedRatings.append((movie, predict))

        for movie_Index in self.seenMovies:
            self.user_scores[movie_Index] = -10 ** 8

        predictedRatings = sorted(range(len(self.user_scores)), key=lambda k: self.user_scores[k])

        recommendation1 = predictedRatings[-1]
        recommendation2 = predictedRatings[-2]
        recommendation3 = predictedRatings[-3]
        return "first " + self.titles[recommendation1][0] + self.titles[recommendation1][1] + " second " + \
               self.titles[recommendation2][0] + self.titles[recommendation2][1] + " third " + \
               self.titles[recommendation3][0] + self.titles[recommendation3][1]

    def stripMovieTitle(self, movie_title):
        result = re.findall(r"(.+) \(\d{4}\)")
        if len(result) > 0:
            return result[0]
        else:
            return movie_title


    #############################################################################
    # 4. Debug info                                                             #
    #############################################################################

    def debug(self, input):
        """Returns debug information as a string for the input string from the REPL"""
        # Pass the debug information that you may think is important for your
        # evaluators
        debug_info = 'debug info'
        return debug_info

    #############################################################################
    # 5. Write a description for your chatbot here!                             #
    #############################################################################
    def intro(self):
        return """
      Your task is to implement the chatbot as detailed in the PA6 instructions.
      Remember: in the starter mode, movie names will come in quotation marks and
      expressions of sentiment will be simple!
      Write here the description for your own chatbot!
      """

    #############################################################################
    # Auxiliary methods for the chatbot.                                        #
    #                                                                           #
    # DO NOT CHANGE THE CODE BELOW!                                             #
    #                                                                           #
    #############################################################################

    def bot_name(self):
        return self.name


if __name__ == '__main__':
    Chatbot()

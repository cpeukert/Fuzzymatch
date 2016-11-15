# Fuzzymatch

This is some code to efficiently merge two (very large) data sets without common identifiers by comparing string variables.
Match quality is evaluated based on parameters of a training dataset.
The training data in the repository is about music artist names collected from weekly top chart data and musicbrainz.org.
Example data sets are about book authors, collected from various public sources such as goodreads.com.

The procedure uses a minimum amount of indexing to reduce the consideration set, based on sub-strings (i.e. first letter).
If that's not what you need, you can think of other substrings, e.g. n-grams, or functions of (sub-)strings, e.g. phonetic algorithms such as Soundex.
The script uses multiple metrics for pair comparison (i.e. distance between strings, phonetics), building on fuzzywuzzy (https://github.com/seatgeek/fuzzywuzzy).
Pair evaluation is aided by parameters obtained in a logistic regression on a manually coded training data set (”supervised learning”).
The procedure uses parallel computing to enable fast and efficient matching of large data sets.
The output file is a list of the best matches between two observations along with a probability score, which can later be used by the researcher to make a decision on the best match.

Feel free to contact me if you have any questions or comments!

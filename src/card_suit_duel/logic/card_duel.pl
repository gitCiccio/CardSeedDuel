% CardSeedDuel - symbolic rules for Approach A (DeepProbLog).
%
% The only learned component is the neural predicate suit/2: SuitNet maps one
% card image to a distribution over the three game suits. Everything below the
% neural predicate is known background knowledge and is never learned.
%
% Atom names must match SUIT_DOMAIN and OUTCOME_DOMAIN in
% card_suit_duel.data.mapping.mappings, in the same order as SUIT_CLASSES.

nn(suit_net, [X], S, [hearts, clubs, spades]) :: suit(X, S).

% Cyclic dominance: hearts > clubs > spades > hearts.
beats(hearts, clubs).
beats(clubs, spades).
beats(spades, hearts).

% Outcome from the first player's perspective, given two suits.
outcome(S, S, draw).
outcome(S1, S2, p1_win) :- beats(S1, S2).
outcome(S1, S2, p2_win) :- beats(S2, S1).

% A duel between two card images.
duel(X, Y, O) :- suit(X, S1), suit(Y, S2), outcome(S1, S2, O).

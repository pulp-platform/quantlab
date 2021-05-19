from collections import namedtuple

from .graph import LightweightGraph
from .rules import LightweightRule

from typing import Union


Commit = namedtuple('Commit', ['rho', 'applications'])


class History(object):

    def __init__(self):
        self._undo = []
        self._redo = []

    def push(self, commit: Commit) -> None:
        self._undo.append(commit)
        self._redo.clear()

    def undo(self) -> Union[Commit, None]:

        try:
            last_commit = self._undo.pop()
            self._redo.append(last_commit)
        except IndexError:
            last_commit = None

        return last_commit

    def redo(self) -> Union[Commit, None]:

        try:
            next_commit = self._redo.pop()
            self._undo.append(next_commit)
        except IndexError:
            next_commit = None

        return next_commit

    def clear(self, force=False):

        if not force:
            confirmation = input("This action is not reversible. Are you sure that you want to delete all the history? [yes/NO]")
            force = confirmation.lower() == 'yes'

        if force:
            self._undo.clear()
            self._redo.clear()


class LightweightEditor(object):

    def __init__(self, graph: LightweightGraph):

        self._graph = graph

        self._history    = History()
        self._in_session = False  # put a lock on the history by preventing editing actions
        self._rho        = None   # current ``LightweightRule``

    def startup(self):
        self._in_session = True

    def pause(self):
        self._in_session = False

    def resume(self):
        self._in_session = True

    def shutdown(self):
        self._in_session = False
        self._history.clear(force=True)

    def set_lwr(self, rho: LightweightRule) -> None:

        if self._in_session:
            self._rho = rho
        else:
            print("Rule setting denied: {} object is non in an editing session.".format(self.__class__.__name__))

    def apply(self) -> None:

        if self._in_session:
            try:
                applications = self._rho.apply(self._graph)
                commit = Commit(rho=self._rho, applications=applications)
                self._history.push(commit)

            except AttributeError:
                print("Rule not set: define a rule before issuing a graph editing instruction.")

        else:
            print("Graph editing denied: {} object is non in an editing session.".format(self.__class__.__name__))

    def unapply(self, n: int = 1) -> None:

        if self._in_session:

            for commit_id in range(0, n):
                last_commit = self._history.undo()
                if last_commit is None:
                    print("I tried to unapply {} commits, but history contained just {}: interrupting instruction.".format(n, commit_id))
                    break
                else:
                    last_commit.rho.unapply(self._graph, last_commit.applications)

        else:
            print("Graph editing denied: {} object is non in an editing session.".format(self.__class__.__name__))

    def reapply(self, n: int = 1) -> None:

        if self._in_session:
            for commit_id in range(0, n):
                next_commit = self._history.redo()
                if next_commit is None:
                    print("I tried to reapply {} commits, but history contained just {}: interrupting instruction.".format(n, commit_id))
                    break
                else:
                    next_commit.rho.apply(self._graph)

        else:
            print("Graph editing denied: {} object is non in an editing session.".format(self.__class__.__name__))

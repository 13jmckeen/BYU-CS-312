#!/usr/bin/python3

from which_pyqt import PYQT_VER
if PYQT_VER == 'PYQT5':
	from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
	from PyQt4.QtCore import QLineF, QPointF
else:
	raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))

import math
import time
import random

# Used to compute the bandwidth for banded version
MAXINDELS = 3

# Used to implement Needleman-Wunsch scoring
MATCH = -3
INDEL = 5
SUB = 1

class GeneSequencing:

	def __init__( self ):
		pass

# This is the method called by the GUI.  _seq1_ and _seq2_ are two sequences to be aligned, _banded_ is a boolean that tells
# you whether you should compute a banded alignment or full alignment, and _align_length_ tells you
# how many base pairs to use in computing the alignment

	def align(self, sequences, table, banded, align_length):
		self.banded = banded
		self.MaxCharactersToAlign = align_length
		results = []

		for i in range(len(sequences)):
			jresults = []
			for j in range(len(sequences)):
				if j < i:
					s = {}
				else:
					a = "-" + sequences[i][:align_length]
					b = "-" + sequences[j][:align_length]
					if banded:
						if abs(len(a) - len(b)) > MAXINDELS:
							score = math.inf
							alignment1 = alignment2 = "No Alignment Possible"
						else:
							if len(b) < len(a):
								score, alignment1, alignment2 = self.banded_algorithm(b, a, align_length)
							else:
								score, alignment1, alignment2 = self.banded_algorithm(a, b, align_length)
					else:
						score, alignment1, alignment2 = self.unrestricted(a, b, align_length)

					s = {'align_cost': score, 'seqi_first100': alignment1, 'seqj_first100': alignment2}
					table.item(i, j).setText('{}'.format(int(score) if score != math.inf else score))
					table.repaint()
					if i == 2 and j == 9:
						val = "for banded" if banded else "for unrestricted"
						print("sequences 3 and 10 " + val)
						print(alignment1[:100])
						print(alignment2[:100])
				jresults.append(s)
			results.append(jresults)
		return results

	def unrestricted(self, a, b, align_length):  # a is vertical, along the side and b is horizontal along the top

		table = []

		a_end = min(len(a), align_length + 1)
		b_end = min(len(b), align_length + 1)

		for i in range(a_end):
			row = []
			for j in range(b_end):
				if i == 0:
					value = (j * INDEL, "left")
				elif j == 0:
					value = (i * INDEL, "top")
				else:
					diagonal = MATCH if a[i] == b[j] else SUB
					value = self.score_direction(table[i - 1][j][0] + INDEL, row[j - 1][0] + INDEL,
												 table[i - 1][j - 1][0] + diagonal)
				row.append(value)
			table.append(row)
		score = table[-1][-1][0]
		i = a_end - 1
		j = b_end - 1
		alignment1 = ""
		alignment2 = ""
		while i != 0 and j != 0:
			if table[i][j][1] == "top":
				alignment1 = a[i] + alignment1
				alignment2 = "-" + alignment2
				i -= 1
			elif table[i][j][1] == "left":
				alignment1 = "-" + alignment1
				alignment2 = b[j] + alignment2
				j -= 1
			else:
				alignment1 = a[i] + alignment1
				alignment2 = b[j] + alignment2
				i -= 1
				j -= 1

		return score, alignment1, alignment2

	def score_direction(self, top, left, diagonal):
		if top <= left and top <= diagonal:
			return top, "top"
		elif left <= top and left <= diagonal:
			return left, "left"
		else:
			return diagonal, "diagonal"

	def banded_algorithm(self, a, b, align_length):
		table = []
		z = (2 * MAXINDELS) + 1

		for i in range(min(len(a), align_length + 1)):
			row = [(math.inf, "")] * z
			row_start = MAXINDELS - i
			for j in range(max(0, row_start), z):

				if i == 0:
					value = ((j - MAXINDELS) * INDEL, "left")
				elif j == row_start and i < MAXINDELS + 1:
					value = (i * INDEL, "diagonal")
				else:
					if i + j - MAXINDELS >= len(b):
						continue
					top = MATCH if a[i] == b[i + j - MAXINDELS] else SUB
					left = row[j - 1][0] + INDEL if j > 0 else math.inf
					diagonal = table[i - 1][j + 1][0] + INDEL if j < (2 * MAXINDELS) else math.inf
					value = self.score_direction(table[i - 1][j][0] + top, left, diagonal)
				row[j] = value
			table.append(row)
		score_index = (-MAXINDELS - 1) + (len(b) - len(a))
		score = table[-1][score_index][0]

		alignment1 = ""
		alignment2 = ""

		i = min(len(a), align_length - 1) - 1
		j = z + score_index

		while i != 0 or j != MAXINDELS:
			if table[i][j][1] == "top":
				alignment1 = a[i] + alignment1
				alignment2 = b[i + j - MAXINDELS] + alignment2
				i -= 1
			elif table[i][j][1] == "left":
				alignment1 = "-" + alignment1
				alignment2 = b[i + j - MAXINDELS] + alignment2
				j -= 1
			else:
				alignment1 = a[i] + alignment1
				alignment2 = "-" + alignment2
				i -= 1
				j += 1
		return score, alignment1, alignment2


class Filter():
	def __init__(self, ifilename, ofilename):
		self.ifile = open(ifilename, 'r')
		self.ofile = open(ofilename, 'w')

	def filter(self, length_thresh):
		count = 0
		for line in self.ifile:
			pts = line.split(',')[:-1]
			if len(pts) > length_thresh:
				self.ofile.writelines(line)
				count += 1
		print("[INFO] %d trajectories are selected"% count)
		self.ifile.close()
		self.ofile.close()
import csv
import numpy as np
import urllib
import simplejson
from sklearn import mixture
import time
import matplotlib as mpl


def make_ellipses(gmm, ax):
    for n, color in enumerate('rgb'):
        v, w = np.linalg.eigh(gmm._get_covars()[n][:2, :2])
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v *= 9
        ell = mpl.patches.Ellipse(gmm.means_[n, :2], v[0], v[1],
                                  180 + angle, color=color)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)


googleGeocodeUrl = 'http://maps.googleapis.com/maps/api/geocode/json?'

def get_coordinates(query, from_sensor=False):
    query = query.encode('utf-8')
    params = {
        'address': query,
        'sensor': "true" if from_sensor else "false"
    }
    url = googleGeocodeUrl + urllib.urlencode(params)
    json_response = urllib.urlopen(url)
    response = simplejson.loads(json_response.read())
    if response['results']:
        location = response['results'][0]['geometry']['location']
        latitude, longitude = location['lat'], location['lng']
    else:
        latitude, longitude = None, None
        print query, "<no results>"
    return latitude, longitude

def dateToSeconds(date):
	tim = time.strptime(date, '%Y-%m-%d')
	return time.mktime(tim)

gender = ['M','F', 'N', 'U'] #15
party = ['D', 'R', '3', 'L', 'I', 'U'] #34 27
typeThing = ['C', 'I', 'P'] #28
seatResult = ['W', 'L'] #41
seatStatus = ['I', 'C', 'O'] #40
transactionType = ['24k', '24z', '15', '24i', '22y', '15j', '24t', '15c', '10', '15e', '11', '19', '24e', '24a', '24f', '11j', '24c', '24n', '20y', '24p', '10j', '18j', '16c', '18k', '18g', '15z', '24g', '20c', '22z', '16g', '18u', '29', '16f', '24r', '24u', '20f'] #5
seatType = ['federal:house','federal:senate', 'federal:president'] #38 39
district = ['CO-05', 'GA-11', 'MI-01', 'WI-05', 'NV-01', 'MI-04', 'CA-21', 'OH-08', 'FL-22', 'VA-01', 'OK-03', 'NY-01', 'CA-29', 'TX-08', 'TX-17', 'MD-08', 'TX-24', 'MD-06', 'NJ-06', 'OR-02', 'MS-02', 'MD-02', 'CA-49', 'IL-06', 'NY-24', 'IL-13', 'MN-03', 'NJ-05', 'AL-05', 'IL-17', 'CT-01', 'GA-07', 'CA-50', 'MS-04', 'WA-07', 'AZ-06', 'MI-12', 'CA-03', 'CO-07', 'TX-06', 'MN-05', 'NY-19', 'FL-05', 'FL-06', 'OK-02', 'FL-14', 'NC-07', 'CT-05', 'MT-01', 'CA-11', 'IA-02', 'DE-01', 'IL-10', 'AL-02', 'GA-12', 'OH-13', 'MI-06', 'TX-02', 'NY-26', 'LA-06', 'NC-06', 'VT-01', 'MO-08', 'NJ-08', 'VA-07', 'CO-01', 'MD-05', 'CA-19', 'VA-11', 'TN-07', 'MO-02', 'HI-02', 'AL-01', 'TX-30', 'PA-01', 'NM-03', 'CA-01', 'OH-16', 'NC-01', 'NM-01', 'OR-05', 'CA-52', 'TX-22', 'WI-03', 'VA-05', 'IL-11', 'NY-07', 'PA-13', 'CA-22', 'VA-06', 'CA-47', 'TX-31', 'MO-09', 'MO-01', 'ID-01', 'PA-14', 'IA-04', 'NY-29', 'NY-25', 'LA-02', 'LA-01', 'KY-02', 'NH-02', 'ME-01', 'AR-01', 'AZ-08', 'CA-48', 'TN-02', 'AZ-03', 'GA-01', 'KY-05', 'OH-07', 'OR-01', 'CA-25', 'MO-05', 'MA-02', 'IL-01', 'MN-04', 'TX-13', 'NC-05', 'PA-06', 'TX-05', 'FL-15', 'RI-01', 'PA-16', 'IN-07', 'MI-15', 'NY-12', 'VA-02', 'NV-03', 'CA-41', 'TX-03', 'HI-01', 'WV-02', 'WA-05', 'PA-07', 'CA-20', 'NC-08', 'FL-10', 'CT-02', 'PA-18', 'TX-15', 'CA-14', 'IN-03', 'NY-06', 'NJ-07', 'AL-06', 'OH-12', 'OH-01', 'MO-03', 'MI-02', 'SC-03', 'CO-04', 'TX-21', 'IL-18', 'PA-08', 'NC-11', 'SC-01', 'NY-20', 'KY-06', 'CA-31', 'NY-17', 'TX-10', 'KS-02', 'MN-02', 'WA-09', 'WI-07', 'CA-43', 'NV-02', 'NY-02', 'IL-14', 'KS-03', 'IL-02', 'TX-23', 'WI-01', 'TX-20', 'CA-17', 'IN-04', 'WI-08', 'MO-04', 'CA-08', 'OR-03', 'WA-03', 'SD-01', 'CT-04', 'MI-14', 'TX-28', 'NE-03', 'MA-09', 'AZ-01', 'CA-05', 'OH-15', 'FL-18', 'LA-05', 'CA-23', 'IL-19', 'TX-26', 'OH-04', 'AZ-05', 'CO-03', 'CA-40', 'OH-06', 'CA-07', 'NC-10', 'MO-07', 'CA-36', 'AR-02', 'NJ-04', 'GA-03', 'RI-02', 'AR-03', 'TX-11', 'UT-02', 'LA-04', 'PA-02', 'NY-27', 'CA-09', 'CA-42', 'GA-06', 'FL-16', 'MS-01', 'KY-01', 'NJ-09', 'PA-15', 'TX-07', 'IN-02', 'KS-04', 'OH-11', 'TN-04', 'WA-08', 'CA-13', 'NY-14', 'FL-07', 'SC-05', 'MO-06', 'PA-04', 'CA-16', 'NY-03', 'WY-01', 'TN-06', 'GA-04', 'ID-02', 'NJ-11', 'MN-07', 'FL-24', 'AL-03', 'GA-08', 'IN-09', 'FL-12', 'UT-03', 'WV-03', 'AK-01', 'OK-01', 'PA-03', 'PA-12', 'MA-08', 'IL-03', 'NY-23', 'PA-05', 'NY-15', 'MI-10', 'CA-44', 'NY-11', 'FL-25', 'WI-06', 'WI-02', 'TX-19', 'WA-04', 'PA-17', 'CO-06', 'CA-37', 'TX-27', 'MI-08', 'IA-03', 'NY-13', 'WA-02', 'NJ-03', 'CA-15', 'NE-02', 'TX-16', 'NJ-01', 'FL-02', 'NY-22', 'IN-06', 'NC-12', 'NC-04', 'CA-45', 'NE-01', 'CA-39', 'MD-04', 'PA-09', 'FL-11', 'OR-04', 'IL-07', 'MS-03', 'FL-01', 'IN-08', 'AR-04', 'MA-05', 'OH-17', 'MA-04', 'FL-13', 'SC-06', 'MN-08', 'MN-01', 'IN-01', 'CA-10', 'GA-05', 'FL-21', 'NH-01', 'IA-05', 'GA-13', 'VI-00', 'CT-03', 'IL-05', 'SC-04', 'WI-04', 'PA-11', 'NC-09', 'ND-01', 'AZ-04', 'CA-51', 'KS-01', 'WA-06', 'NM-02', 'MN-06', 'OK-04', 'NY-28', 'CA-30', 'SC-02', 'VA-04', 'ME-02', 'TN-03', 'NJ-02', 'WV-01', 'OH-03', 'NC-03', 'GA-09', 'MD-07', 'NY-10', 'FL-20', 'IL-08', 'TX-25', 'FL-08', 'AZ-07', 'GA-02', 'TX-29', 'IA-01', 'FL-19', 'AL-07', 'CA-27', 'VA-09', 'TX-12', 'LA-07', 'MI-03', 'MA-07', 'MI-07', 'TN-05', 'CA-12', 'TX-32', 'PA-10', 'CA-46', 'TN-08', 'CA-28', 'OH-18', 'CA-34', 'IN-05', 'OH-09', 'OK-05', 'FL-04', 'NY-08', 'NJ-12', 'MA-10', 'FL-03', 'TX-09', 'NY-04', 'CA-32', 'CA-24', 'CA-53', 'MI-09', 'CA-26', 'IL-16', 'CA-35', 'AZ-02', 'TX-18', 'OH-02', 'AL-04', 'KY-03', 'FL-23', 'VA-08', 'NC-02', 'UT-01', 'OH-05', 'MI-13', 'CA-38', 'VA-10', 'CA-02', 'MD-01', 'TX-04', 'NY-18', 'IL-09', 'IL-04', 'GA-10', 'PR-00', 'OH-10', 'DC-00', 'TN-09', 'CA-33', 'MA-03', 'FL-17', 'OH-14', 'MD-03', 'NC-13', 'MA-06', 'LA-03', 'NY-09', 'GU-00', 'NY-16', 'TX-14', 'NY-05', 'NJ-13', 'NY-21', 'FL-09', 'MI-05', 'IL-15', 'MP-00', 'TN-01', 'IL-12', 'CO-02', 'VA-03', 'TX-01', 'CA-04', 'AS-00', 'MA-01', 'PA-19', 'CA-06'] #36 37
state = ['CO', 'GA', '', 'MI', 'ME', 'WI', 'NV', 'CA', 'OH', 'FL', 'VA', 'OK', 'NY', 'TX', 'MD', 'MN', 'MA', 'NJ', 'OR', 'MS', 'IL', 'AL', 'CT', 'DE', 'WA', 'AZ', 'RI', 'NC', 'MT', 'IA', 'WV', 'LA', 'VT', 'MO', 'TN', 'HI', 'PA', 'NM', 'ID', 'KY', 'NH', 'AR', 'KS', 'IN', 'WY', 'SD', 'SC', 'UT', 'NE', 'AK', 'VI', 'ND', 'PR', 'DC', 'GU', 'MP', 'AS'] #29 30 18
 
fieldVec = dict([(15, gender), (34, party), (27, party), (28, typeThing), (41, seatResult), (40, seatStatus), (5, transactionType), (38, seatType), (39, seatType), (36, district), (37, district), (29, state), (30, state), (18, state)])

def fieldToNum(input, fields):
	for i in range(len(fields)):
		if(fields[i].lower()==input.lower()):
			return i
	return -1


#print dateToSeconds('2012-02-29')	
#coord = get_coordinates("2157 Ridge Avenue Evanston Il")
#print coord

with open('contributions.fec.2012.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    lines = []
    for row in spamreader:
    	lines.append(row)
    print zip(lines[0],range(len(lines[0])))
    leng = len(lines)
    print leng
    lines = lines[0:leng-1]

    # for index in range(len(lines[0])):
    # 	print index, lines[0][index]
    # 	total = []
    # 	printTot = True
    # 	for row in lines:
    # 		if(row[index] not in total):
    # 			if(len(total)>500):
    # 				print "Too many"
    # 				printTot = False
    # 				break
    # 			else:
    # 				total.append(row[index])
    # 	if printTot:
    # 		print total
    		
    newLines = lines[1:leng-1]
    #print newLines.shape
    result = []
    for i in range(1000): #range(len(newLines)):
    	line = newLines[1]
    	#print i
    	tempLine = []
    	for i in range(len(line)):
    		if i in fieldVec:
    			tempLine.append(fieldToNum(line[i], fieldVec[i]))
    	if (len(line[9]) >0):
    		tempLine.append(dateToSeconds(line[9]))
    	result.append(tempLine)
    #print result
    result = np.array(result)
    print 1

    g = mixture.GMM(n_components=len(result[1]))
    print 2
    g.fit(result)
    print g
    n_classes = len(np.unique(y_train))

# Try GMMs using different types of covariances.
classifiers = dict((covar_type, GMM(n_components=n_classes,
                    covariance_type=covar_type, init_params='wc', n_iter=20))
                   for covar_type in ['spherical', 'diag', 'tied', 'full'])

n_classifiers = len(classifiers)

pl.figure(figsize=(3 * n_classifiers / 2, 6))
pl.subplots_adjust(bottom=.01, top=0.95, hspace=.15, wspace=.05,
                   left=.01, right=.99)


for index, (name, classifier) in enumerate(classifiers.iteritems()):
    # Since we have class labels for the training data, we can
    # initialize the GMM parameters in a supervised manner.
    classifier.means_ = np.array([X_train[y_train == i].mean(axis=0)
                                  for i in xrange(n_classes)])

    # Train the other parameters using the EM algorithm.
    classifier.fit(X_train)

    h = pl.subplot(2, n_classifiers / 2, index + 1)
    make_ellipses(classifier, h)

    for n, color in enumerate('rgb'):
        data = iris.data[iris.target == n]
        pl.scatter(data[:, 0], data[:, 1], 0.8, color=color,
                   label=iris.target_names[n])
    # Plot the test data with crosses
    for n, color in enumerate('rgb'):
        data = X_test[y_test == n]
        pl.plot(data[:, 0], data[:, 1], 'x', color=color)

    y_train_pred = classifier.predict(X_train)
    train_accuracy = np.mean(y_train_pred.ravel() == y_train.ravel()) * 100
    pl.text(0.05, 0.9, 'Train accuracy: %.1f' % train_accuracy,
            transform=h.transAxes)

    y_test_pred = classifier.predict(X_test)
    test_accuracy = np.mean(y_test_pred.ravel() == y_test.ravel()) * 100
    pl.text(0.05, 0.8, 'Test accuracy: %.1f' % test_accuracy,
            transform=h.transAxes)

    pl.xticks(())
    pl.yticks(())
    pl.title(name)

pl.legend(loc='lower right', prop=dict(size=12))
pl.show()
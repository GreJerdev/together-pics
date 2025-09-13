

import cv2
from face_finder import FaceFinder
from pytorch_classifier import PyTorchResNetClassifier

def main():

    pytorch_classifier = PyTorchResNetClassifier()
    face_finder = FaceFinder()
    image = cv2.imread("images (1).jpg")
    face_locations, face_crops = face_finder.detect_faces(image)
    print(face_locations)
    print(face_crops)
    cv2.imshow("Face Finder", image)
    for i in range(len(face_crops)):
        image_tensor = pytorch_classifier.preprocess_cv2_image(face_crops[i])
        class_labels = pytorch_classifier.classify_image(image_tensor)

        print(i, class_labels)
        cv2.imshow("Face Finder", face_crops[i])
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


'''
0 [(906, 0.056783296167850494, 'Class_906'), (901, 0.03313130885362625, 'Class_901'), (439, 0.02746736817061901, 'Class_439'), (633, 0.027322370558977127, 'Class_633'), (653, 0.026228275150060654, 'Class_653')]
1 [(473, 0.09010489284992218, 'Class_473'), (631, 0.0651613101363182, 'Class_631'), (906, 0.05034219101071358, 'Class_906'), (813, 0.038810089230537415, 'Class_813'), (772, 0.03643724322319031, 'Class_772')]
2 [(147, 0.1483369767665863, 'Class_147'), (149, 0.0806073546409607, 'Class_149'), (371, 0.05283907428383827, 'Class_371'), (279, 0.04238668829202652, 'Class_279'), (335, 0.03784754127264023, 'Class_335')]

0 [(906, 0.056783296167850494, 'Class_906'), (901, 0.03313130885362625, 'Class_901'), (439, 0.02746736817061901, 'Class_439'), (633, 0.027322370558977127, 'Class_633'), (653, 0.026228275150060654, 'Class_653')]
1 [(473, 0.09010489284992218, 'Class_473'), (631, 0.0651613101363182, 'Class_631'), (906, 0.05034219101071358, 'Class_906'), (813, 0.038810089230537415, 'Class_813'), (772, 0.03643724322319031, 'Class_772')]
2 [(147, 0.1483369767665863, 'Class_147'), (149, 0.0806073546409607, 'Class_149'), (371, 0.05283907428383827, 'Class_371'), (279, 0.04238668829202652, 'Class_279'), (335, 0.03784754127264023, 'Class_335')]

'''









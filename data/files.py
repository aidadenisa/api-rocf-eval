
import os
import base64

def saveROCFImage(imageb64, uploadPath, doctorID, patientCode, datetime):
        date = datetime.strftime("%d:%m:%Y-%H:%M:%S")
        # filename = secure_filename(imageb64.filename)
        filename = patientCode + "_" + date
        doctorFolderPath = os.path.join(uploadPath, doctorID)

        if filename != '':
            image = base64.b64decode(imageb64)
            # file_ext = os.path.splitext(filename)[1]
            # if file_ext not in app.config['UPLOAD_EXTENSIONS']:
            #     abort(400)
            
            if not os.path.isdir(doctorFolderPath):
                os.mkdir(os.path.join(doctorFolderPath))

            location = os.path.join(doctorFolderPath, filename + ".png")
            
            with open(location, "wb") as fh:
                fh.write(image)
                        
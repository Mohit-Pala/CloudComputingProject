getImages() {
    this.fs.returnAllImages().then((res) => {
        this.previousImages = res;
    }).catch((error) => {
        console.log("Error getting images", error);
    });
}

onSubmit() {
    console.log('File submitted:', this.imagePreview);
    if (this.imagePreview) {
        this.fs.putImage(this.imagePreview).then(() => {
            console.log("Image uploaded successfully");
        }).catch((error) => {
            console.log("Error uploading image", error);
        });
    }
}


async putImage(base64Image: string) {
    const collectionRef = collection(this.firestore, "images")
    const docRef = await addDoc(collectionRef, {
        image: base64Image
    }).then((res) => {
        console.log("Document written", res)
    }
    ).catch((error) => {
        console.log("Error adding document", error)
    })
}

  async returnAllImages() {
    const collectionRef = collection(this.firestore, "images")
    const querySnapshot = await getDocs(collectionRef)
    if (querySnapshot.empty) {
        console.log("No documents found!")
        return []
    }

    const images: string[] = []
    querySnapshot.forEach((doc) => {
        images.push(doc.data()['image'])
    })
    return images
}
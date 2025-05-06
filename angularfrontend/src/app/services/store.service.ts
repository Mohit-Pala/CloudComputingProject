import { Injectable } from '@angular/core';
import { initializeApp } from '@angular/fire/app';
import { addDoc, collection, Firestore, getDocs, getFirestore } from '@angular/fire/firestore';
import { firebaseConfig } from '../../../api_keys';

@Injectable({
  providedIn: 'root'
})
export class StoreService {

  private firestore: Firestore;

  constructor() {
    const app = initializeApp(firebaseConfig)
    this.firestore = getFirestore(app)
  }

  async getAllImages(): Promise<string[] | null> {
    const imageRef = collection(this.firestore, 'images')
    const querySnapshot = await getDocs(imageRef)
    if (querySnapshot.empty) {
      console.log("No documents found!")
      return null
    }
    let imageBase64: string[] = []
    querySnapshot.docs.forEach((doc) => {
      const data = doc.data()
      imageBase64.push(data['image'])
    })
    return imageBase64
  }

  async putImage(image: string): Promise<void> {
    const imageRef = collection(this.firestore, 'images')
    const docRef = await addDoc(imageRef, {
      image: image
    })
    console.log("Document written with ID: ", docRef.id)
  }
}

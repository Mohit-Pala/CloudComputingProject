import { CommonModule } from '@angular/common';
import { Component, inject, OnInit } from '@angular/core';
import { StoreService } from '../services/store.service';

@Component({
  selector: 'app-gallery',
  imports: [CommonModule],
  templateUrl: './gallery.component.html'
})
export class GalleryComponent implements OnInit {

  images: string[] = Array(4).fill(null)

  firestore = inject(StoreService)

  ngOnInit() {
    this.firestore.getAllImages().then((res) => {
      if (res) {
        this.images = res
      } else {
        console.log("No images found")
      }
    }).catch((err) => {
      console.error("Error fetching images:", err)
    })
  }
}

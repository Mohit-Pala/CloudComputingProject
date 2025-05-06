import { Component, inject } from '@angular/core';
import { GenService } from '../services/gen/gen.service';
import { CommonModule } from '@angular/common';
import { StoreService } from '../services/store.service';

@Component({
  selector: 'app-main',
  imports: [CommonModule],
  templateUrl: './main.component.html'
})
export class MainComponent {
  isGenerated = false
  generatedImageTag: string = ''
  firestore = inject(StoreService)

  constructor(private genService: GenService) { }

  async onGenerate() {
    try {
      this.generatedImageTag = await this.genService.generateImage()
      this.isGenerated = true
    } catch (error) {
      console.error('Error generating image:', error)
    }
  }

  onStore() {
    if (!this.generatedImageTag) {
      console.error("No image to store")
      return
    }

    if (this.generatedImageTag.length > 1000000) {
      console.error("Image is too large to store")
      return
    }

    if (this.generatedImageTag.length < 100) {
      alert("Image not generated yet")
      return
    }

    this.firestore.putImage(this.generatedImageTag).then((res) => {
      console.log("Image stored successfully")
      console.log(res)
    }).catch((err) => {
      console.error("Error storing image:", err)
    })
  }
}

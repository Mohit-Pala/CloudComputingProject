import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';

@Injectable({
  providedIn: 'root'
})
export class GenService {
  private apiUrl = 'https://img-gen-964405567549.us-east1.run.app/'


  async generateImage(): Promise<string> {
    const response = await fetch(this.apiUrl)
    const imageTag = await response.text()
    return imageTag
  }

}

import { Routes } from '@angular/router';
import { MainComponent } from './main/main.component';
import { GalleryComponent } from './gallery/gallery.component';

export const routes: Routes = [
    {path: '', component: MainComponent},
    {path: 'gallery', component: GalleryComponent}
];

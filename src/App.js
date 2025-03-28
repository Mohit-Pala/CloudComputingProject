import React, { useState } from 'react';
import cat from './cat7.png';
import './App.css';

function App() {
  const [inputValue, setInputValue] = useState('');

  const handleChange = (event) => {
    setInputValue(event.target.value);
  };

  const handleSubmit = (event) => {
    event.preventDefault();
    alert(`Submitted: ${inputValue}`);
    setInputValue('');
  };

  return (
    <div className='App-header'>
      <div id="image-box">
        <img src={cat} alt="" width="128" height="128"></img>
      </div>
      <div id="input">
        <form onSubmit={handleSubmit}>
          <label for="options">Select prompt for image</label>
          <br></br>
          <select
            name="options"
            value={inputValue}
            onChange={handleChange}
            className='select-box'
          >
            <option value="airplane">airplane</option>
            <option value="automobile">automobile</option>
            <option value="bird">bird</option>
            <option value="cat">cat</option>
            <option value="deer">deer</option>
            <option value="dog">dog</option>
            <option value="frog">frog</option>
            <option value="horse">horse</option>
            <option value="ship">ship</option>
            <option value="truck">truck</option>
          </select>
          <button type="submit"className='select-button'>Submit</button>
      </form>
      </div>
    </div>
  );
}

export default App;

import React from "react";
import { Service } from "../service/api";
import { Jumbotron, Form, Button, Alert, Card, Table, Accordion } from "react-bootstrap";
import './Extract.css';
import { base_url } from "../config"
import axios from "axios";


function downloadObjectAsJson(exportObj, exportName) {
  var dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(exportObj));
  var downloadAnchorNode = document.createElement('a');
  downloadAnchorNode.setAttribute("href", dataStr);
  downloadAnchorNode.setAttribute("download", exportName + ".json");
  document.body.appendChild(downloadAnchorNode); // required for firefox
  downloadAnchorNode.click();
  downloadAnchorNode.remove();
}

export class Extract extends React.Component {
  service = null;

  colorMap = {
    "No Concept": "white",
    "Product": "#ff5454",
    "Reactants": "#f0ba84",
    "Reaction": "#723d46",
    "Solvent": "#eb5e28",
    "Catalyst_Reagents": "#54ffc3",
    "Yield": "#ffc300",
    "Temperature": "#3cbab1",
    "Time": "#d8d243",
  }

  constructor() {
    super();
    this.demoParagraph = 'We were excited to find that , with 2.0 equiv of copper acetate and DMSO as the solvent , 2-(3-(methylthio)naphthalen-2-yl)pyridine was obtained as a single product in 89 % yield at 125 °C ( Table 1 ) .'
    let defaultParagraph = "";
    this.state = { showStartSuccess: false, isInputText: true, inputText: defaultParagraph, showExtraction: false, tokens: [], reactions: [], selectedReaction: 0, selectedFiles: null, documentInProgress: false, jsonOutput: null };
    this.service = new Service();
    this.createExtraction = this.createExtraction.bind(this)
    this.downloadExtractionAsJson = this.downloadExtractionAsJson.bind(this)
  }

  //TODO: separate upload logic into service
  onFileChange = (event) => {
    // Update the state
    this.setState({ selectedFiles: event.target.files });
  };

  // On file upload (click the upload button)
  onFileUpload = () => {
    // Create an object of formData
    const formData = new FormData();

    // Update the formData object

    for (var i = 0; i < this.state.selectedFiles.length; i++) {
      let file = this.state.selectedFiles[i];
      let fileName = "file_" + String(i);
      formData.append(fileName, file, file.name);
    }

    // Send formData object
    const api_endpoint = base_url + "/extractDocument";

    console.log(api_endpoint)

    // banner to say it was succesfully uploaded
    this.setState({ showStartSuccess: true, documentInProgress: true }, () => {
      window.setTimeout(() => {
        this.setState({ showStartSuccess: false });
      }, 2000);
    });

    axios.post(api_endpoint, formData).then((res) => {
      console.log(res)
      this.setState({ jsonOutput: res.data.extractions, documentInProgress: false })
    });
  };

  handleSubmit = () => {
    return false;
  };

  downloadExtractionAsJson() {
    downloadObjectAsJson(this.state.jsonOutput, 'extraction')
    window.setTimeout(() => {
      this.setState({ jsonOutput: null })
    }, 20000)
  }

  createExtraction() {
    let text = this.state.inputText;

    // Show start banner
    this.setState({ showStartSuccess: true, showExtraction: false, selectedReaction: 0 })
    setTimeout(() => {
      this.setState({ showStartSuccess: false })
    }, 2000);

    this.service.extractParagraph(text, (res) => {
      const tokens = res.data.extractions[0].tokens
      const reactions = res.data.extractions[0].reactions
      this.setState({ showExtraction: true, tokens: tokens, reactions: reactions })
      this.render()
    })
  }

  renderReaction(tokens, reaction) {
    let toks = []
    let ent = []

    for (let j = 0; j < tokens.length; j++) {
      ent.push('No Concept')
    }

    for (let key of Object.keys(reaction)) {
      if (key === "Product") {
        const locs = reaction[key]
        const start = locs[0]
        const stop = locs[1]

        for (let j = start; j <= stop; j++) {
          ent[j] = key
        }
      } else {
        for (let j = 0; j < reaction[key].length; j++) {
          const locs = reaction[key][j]
          const start = locs[0]
          const stop = locs[1]

          for (let i = start; i <= stop; i++) {
            ent[i] = key
          }
        }
      }
    }

    let i = 0;
    while (i < ent.length) {

      let j = i + 1;
      while (ent[i] === ent[j]) {
        j++;
      };

      const entToks = tokens.slice(i, j);
      const entText = entToks.join(' ')

      const style = {
        cursor: 'help',
        backgroundColor: 'white',
        borderColor: this.colorMap[ent[i]],
        borderWidth: '2px',
      }

      if (ent[i] === "No Concept") {
        toks.push(` ${entText} `)
      } else {
        toks.push(<button className="btn btn-sm" data-toggle="tooltip" data-placement="top" title={ent[i]} style={style} data-original-title={ent[i]} key={i}><span>{entText}</span></button>)
      }
      i = j
    }

    return <div>{toks}</div>
  }

  renderReactionSelector() {

    const reactionOptions = []
    for (let j = 0; j < this.state.reactions.length; j++) {
      reactionOptions.push(<option key={j} data-key={j}>{`Reaction ${j}`}</option>)
    }

    return <Form.Group>
      <Form.Control as="select" onChange={(e) => {
        const selectedIndex = e.target.options.selectedIndex;
        const key = e.target.options[selectedIndex].getAttribute('data-key')
        this.setState({
          selectedReaction: key
        })
      }}>
        {reactionOptions}
      </Form.Control>
    </Form.Group>
  }

  renterExtractions() {
    if (this.state.showExtraction) {
      return (<Card>
        <Card.Header as="h5">Extraction</Card.Header>
        {this.renderReactionSelector()}
        <Card.Body>
          {this.renderReaction(this.state.tokens, this.state.reactions[this.state.selectedReaction])}
        </Card.Body>
      </Card>)
    } else {
      return <></>
    }
  }

  renderColorLegend() {
    return <>
      <Accordion>
        <Card>
          <Card.Header>
            <Accordion.Toggle as={Button} variant="link" eventKey="0">
              Color Key
            </Accordion.Toggle>
          </Card.Header>
          <Accordion.Collapse eventKey="0">
            <Table striped bordered hover size="small">
              <thead>
                <tr>
                  <th>Entity Name</th>
                  <th>Color</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(this.colorMap).map((entry) => {
                  const key = entry[0];
                  const col = entry[1];
                  return <tr>
                    <td>{key}</td>
                    <td style={{
                      backgroundColor: entry[1],
                    }}></td>
                  </tr>
                })}
              </tbody>
            </Table>
          </Accordion.Collapse>
        </Card>
      </Accordion>

    </>
  }

  // File content to be displayed after

  render() {

    let cname = 'pt-5' + ((this.state.documentInProgress || this.state.jsonOutput !== null) ? '' : ' d-none')
    console.log('inpgrogress', this.state.documentInProgress)
    console.log('jsonOutput', this.state.jsonOutput)
    console.log('className: ', cname)

    return (
      <>
        <Alert variant="success" show={this.state.showStartSuccess}>
          Extraction succesfully started! This might take a few minutes.
        </Alert>
        <Jumbotron>
          <h4>Online Demo</h4>

          <div className="button-box">
            <button className={`${this.state.isInputText ? 'selected' : ''}`} onClick={() => {
              this.setState({ isInputText: true })
            }}>Text Input</button>
            &nbsp;&nbsp;
            <button className={`${this.state.isInputText ? '' : 'selected'}`} onClick={() => {
              this.setState({ isInputText: false })
            }}>Upload File</button></div>
          <br />

          <div className={"textInput" + (this.state.isInputText ? '' : ' d-none')}>
            <Form>
              <Form.Group>
                <Form.Control as="textarea" placeholder="Hello" rows={5} value={this.state.inputText} onChange={(e) => this.setState({ inputText: e.target.value })} placeholder="Type or paste some text here."></Form.Control>

                <p><span className="font-weight-bold">Example:</span> {this.demoParagraph}</p>
                <br />
                <Button variant="primary" onClick={this.createExtraction}>Extract</Button>
              </Form.Group>
            </Form>

            {this.renderColorLegend()}
            {this.renterExtractions()}
          </div>

          <div className={"fileInput" + (this.state.isInputText ? ' d-none' : '')}>
            <p>Currently works with PDF version of journal articles published by ACS</p>
            <Form onSubmit={this.handleSubmit}>
              <Form.Group>
                <Form.File
                  id="uploadedFiles"
                  onChange={this.onFileChange}
                />
              </Form.Group>
            </Form>

            <Button onClick={this.onFileUpload} >Submit</Button>

            <div className={cname}>
              <span>An extraction is currently getting generated. If it is not currently visible, the download button should appear shortly.</span> <br />
              <Button type="secondary" className={this.state.jsonOutput !== null ? '' : 'd-none'} onClick={this.downloadExtractionAsJson}> Download Extraction </Button>
            </div>
          </div>

        </Jumbotron >
      </>
    );
  }
}
